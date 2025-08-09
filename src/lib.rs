use anyhow::{Context, Result};
use num_complex::Complex32;
use rustfft::FftPlanner;
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::Semaphore;

#[derive(Debug, Serialize)]
pub struct CorrelationResult {
    pub file: String,
    pub offset_seconds: f64,
}

/// Paramètres (à ajuster)
const BLOCK_SIZE_SAMPLES: usize = 131_072; // taille bloc tentative (échantillons)
const MIN_N_CAP: usize = 1 << 14; // 16k minimum FFT size
const ABS_MAX_N_CAP: usize = 1 << 26; // safety hard cap (~67M)
const USABLE_MEM_PERCENT: usize = 85; // utiliser 85% de la mémoire disponible

/// Probe sample rate and duration (seconds) using ffprobe.
/// Returns (sample_rate, duration_seconds)
async fn probe_samplerate_duration(path: &Path) -> Result<(u32, f64)> {
    let out = Command::new("ffprobe")
        .args(&[
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.to_string_lossy().as_ref(),
        ])
        .output()
        .await
        .with_context(|| format!("ffprobe failed for {:?}", path))?;

    if !out.status.success() {
        anyhow::bail!("ffprobe returned non-zero for {:?}", path);
    }
    let txt = String::from_utf8_lossy(&out.stdout);
    let mut sr: Option<u32> = None;
    let mut dur: Option<f64> = None;
    for line in txt.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if sr.is_none() && s.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(v) = s.parse::<u32>() {
                sr = Some(v);
                continue;
            }
        }
        if dur.is_none() {
            if let Ok(v) = s.parse::<f64>() {
                dur = Some(v);
                continue;
            }
        }
    }
    let sr = sr.unwrap_or(48000);
    let dur = dur.unwrap_or(0.0);
    Ok((sr, dur))
}

/// cgroup-aware detection: try to detect container memory limit and available memory.
/// Returns available bytes (conservative).
fn detect_cgroup_memory_limit_bytes() -> Option<usize> {
    // try cgroup v2 common file
    let cg_v2_root = Path::new("/sys/fs/cgroup/memory.max");
    if cg_v2_root.exists() {
        if let Ok(s) = fs::read_to_string(cg_v2_root) {
            let s = s.trim();
            if s != "max" {
                if let Ok(v) = s.parse::<u128>() {
                    return Some(v as usize);
                }
            } else {
                return None;
            }
        }
    }

    // parse /proc/self/cgroup
    let cgroup_txt = fs::read_to_string("/proc/self/cgroup").ok()?;
    for line in cgroup_txt.lines() {
        if line.starts_with("0::") {
            if let Some(cpath) = line.splitn(3, ':').nth(2) {
                let cpath = if cpath.starts_with('/') { &cpath[1..] } else { cpath };
                let candidate = Path::new("/sys/fs/cgroup").join(cpath).join("memory.max");
                if candidate.exists() {
                    if let Ok(s) = fs::read_to_string(&candidate) {
                        let s = s.trim();
                        if s == "max" {
                            return None;
                        }
                        if let Ok(v) = s.parse::<u128>() {
                            return Some(v as usize);
                        }
                    }
                }
                // fallback
                let alt = Path::new("/sys/fs/cgroup").join("memory.max");
                if alt.exists() {
                    if let Ok(s) = fs::read_to_string(&alt) {
                        let s = s.trim();
                        if s == "max" {
                            return None;
                        }
                        if let Ok(v) = s.parse::<u128>() {
                            return Some(v as usize);
                        }
                    }
                }
            }
        }
    }

    // cgroup v1: look for controllers containing "memory"
    for line in cgroup_txt.lines() {
        let parts: Vec<&str> = line.splitn(3, ':').collect();
        if parts.len() == 3 {
            let controllers = parts[1];
            let cpath = parts[2];
            if controllers.split(',').any(|c| c == "memory") {
                let cpath = if cpath.starts_with('/') { &cpath[1..] } else { cpath };
                let candidate = Path::new("/sys/fs/cgroup/memory").join(cpath).join("memory.limit_in_bytes");
                if candidate.exists() {
                    if let Ok(s) = fs::read_to_string(&candidate) {
                        let s = s.trim();
                        if let Ok(v) = s.parse::<u128>() {
                            if v > (1u128 << 62) {
                                return None;
                            }
                            return Some(v as usize);
                        }
                    }
                }
                let alt = Path::new("/sys/fs/cgroup").join("memory").join("memory.limit_in_bytes");
                if alt.exists() {
                    if let Ok(s) = fs::read_to_string(&alt) {
                        let s = s.trim();
                        if let Ok(v) = s.parse::<u128>() {
                            if v > (1u128 << 62) {
                                return None;
                            }
                            return Some(v as usize);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Read MemAvailable from /proc/meminfo in bytes as fallback.
/// If fails, returns a conservative default (512 MB).
fn read_mem_available_bytes_fallback() -> usize {
    const DEFAULT: usize = 512 * 1024 * 1024;
    match fs::read_to_string("/proc/meminfo") {
        Ok(s) => {
            for line in s.lines() {
                if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
            for line in s.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
            DEFAULT
        }
        Err(_) => DEFAULT,
    }
}

/// Final detection: prefer cgroup limit if present and sensible,
/// otherwise fall back to MemAvailable.
fn detect_available_memory_bytes() -> usize {
    let meminfo_avail = read_mem_available_bytes_fallback();

    if let Some(cg_limit) = detect_cgroup_memory_limit_bytes() {
        std::cmp::min(cg_limit, meminfo_avail)
    } else {
        meminfo_avail
    }
}

/// Compute a safe max N cap (power of two) based on available memory and desired usage percent.
fn compute_max_n_cap() -> usize {
    let avail = detect_available_memory_bytes();
    let usable = avail.saturating_mul(USABLE_MEM_PERCENT) / 100;
    // estimate bytes per complex element for two complex buffers (ref_fft + in_buf) + headroom
    let bytes_per_element = 16usize; // Complex32 (8 bytes) * 2 = 16
    if usable <= bytes_per_element {
        return MIN_N_CAP;
    }
    let mut n = usable / bytes_per_element;
    if n < MIN_N_CAP {
        n = MIN_N_CAP;
    }
    if n > ABS_MAX_N_CAP {
        n = ABS_MAX_N_CAP;
    }
    // round down to nearest power of two
    let p = next_pow2_floor(n);
    if p < MIN_N_CAP {
        MIN_N_CAP
    } else {
        p
    }
}

/// next power of two (>= n)
fn next_pow2(mut n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if std::mem::size_of::<usize>() > 4 {
        n |= n >> 32;
    }
    n + 1
}

/// largest power of two <= n
fn next_pow2_floor(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if n.is_power_of_two() {
        return n;
    }
    let mut p = 1usize;
    while p <= n {
        p <<= 1;
    }
    p >> 1
}

/// Spawn ffmpeg child that outputs mono pcm_f32le to stdout.
fn spawn_ffmpeg_child(path: &Path, pan_expr: Option<String>, target_sr: Option<u32>) -> Result<Child> {
    let mut args: Vec<String> = vec![
        "-vn".to_string(),
        "-nostdin".to_string(),
        "-i".to_string(),
        path.to_string_lossy().into_owned(),
    ];

    if let Some(pan) = pan_expr {
        args.push("-af".to_string());
        args.push(pan);
    }

    args.push("-ac".to_string());
    args.push("1".to_string());

    if let Some(sr) = target_sr {
        args.push("-ar".to_string());
        args.push(format!("{}", sr));
    }

    args.extend(vec![
        "-f".to_string(),
        "f32le".to_string(),
        "-acodec".to_string(),
        "pcm_f32le".to_string(),
        "-".to_string(),
    ]);

    let mut cmd = Command::new("ffmpeg");
    cmd.args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);

    let child = cmd.spawn().context("failed to spawn ffmpeg")?;
    Ok(child)
}

/// Read the whole stdout of ffmpeg into Vec<f32> (mono pcm_f32le).
async fn read_all_pcm_f32_from_ffmpeg(path: &Path, pan_expr: Option<String>, target_sr: Option<u32>) -> Result<(u32, Vec<f32>)> {
    let (sr0, _dur) = probe_samplerate_duration(path).await?;
    let target_sr = target_sr.unwrap_or(sr0);

    let mut args: Vec<String> = vec![
        "-vn".to_string(),
        "-nostdin".to_string(),
        "-i".to_string(),
        path.to_string_lossy().into_owned(),
    ];

    if let Some(p) = pan_expr {
        args.push("-af".to_string());
        args.push(p);
    }

    args.push("-ac".to_string());
    args.push("1".to_string());
    args.push("-ar".to_string());
    args.push(format!("{}", target_sr));
    args.extend(vec![
        "-f".to_string(),
        "f32le".to_string(),
        "-acodec".to_string(),
        "pcm_f32le".to_string(),
        "-".to_string(),
    ]);

    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("spawn ffmpeg for read_all")?;

    let stdout = child.stdout.take().context("no stdout from ffmpeg")?;
    let mut reader = tokio::io::BufReader::new(stdout);

    let mut buf: Vec<u8> = Vec::new();
    reader.read_to_end(&mut buf).await.context("read ffmpeg stdout")?;

    let _ = child.wait().await;

    let mut samples = Vec::with_capacity(buf.len() / 4);
    for chunk in buf.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        samples.push(f32::from_le_bytes(bytes));
    }
    Ok((target_sr, samples))
}

/// Core: streaming cross-correlation using overlap-save with memory-controlled FFT size and buffer reuse.
/// Loads the shorter file fully (reference), precomputes its FFT once at chosen N, then streams the other file in blocks sized B = N - m + 1.
pub async fn second_correlation_streaming(in1: &str, in2: &str, pool_capacity: usize) -> Result<CorrelationResult> {
    let max_n_cap = compute_max_n_cap();

    let p1 = Path::new(in1);
    let p2 = Path::new(in2);
    let (sr1, dur1) = probe_samplerate_duration(p1).await?;
    let (sr2, dur2) = probe_samplerate_duration(p2).await?;

    let samples1 = (sr1 as f64 * dur1).round() as usize;
    let samples2 = (sr2 as f64 * dur2).round() as usize;

    let (ref_path, stream_path, _ref_samples_est, _sr_ref, _sr_stream) = if samples1 <= samples2 {
        (p1, p2, samples1, sr1, sr2)
    } else {
        (p2, p1, samples2, sr2, sr1)
    };

    let target_sr = std::cmp::min(sr1, sr2);
    let pan_expr: Option<String> = None;

    let (_sr_used, mut ref_samples) =
        read_all_pcm_f32_from_ffmpeg(ref_path, pan_expr.clone(), Some(target_sr)).await?;

    if ref_samples.is_empty() {
        anyhow::bail!("reference samples empty");
    }

    ref_samples.reverse();
    let m = ref_samples.len();

    if m > max_n_cap {
        anyhow::bail!(
            "reference too large for in-memory FFT approach (m = {}, max_n_cap = {}). \
            Consider increasing available memory or using a segmented algorithm.",
            m,
            max_n_cap
        );
    }

    // choose N to try to get B ~= BLOCK_SIZE_SAMPLES
    let desired_n = next_pow2(m.saturating_sub(1) + BLOCK_SIZE_SAMPLES);
    let mut n = if desired_n <= max_n_cap {
        desired_n
    } else {
        let cand = next_pow2_floor(max_n_cap);
        if cand < m {
            let cand2 = next_pow2(m);
            if cand2 > max_n_cap {
                anyhow::bail!(
                    "cannot find fft size >= m within cap (m={}, max_n_cap={})",
                    m,
                    max_n_cap
                );
            }
            cand2
        } else {
            cand
        }
    };

    if n < m {
        n = next_pow2(m);
        if n > max_n_cap {
            anyhow::bail!(
                "computed FFT size exceeds cap after adjust (m={}, n={}, cap={})",
                m,
                n,
                max_n_cap
            );
        }
    }

    let block_size_b = n.saturating_sub(m).saturating_add(1); // B = N - m + 1
    if block_size_b == 0 {
        anyhow::bail!("computed block size is zero (n={}, m={})", n, m);
    }

    // planner and precompute ref FFT at size n
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut ref_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..m.min(n) {
        ref_buf[i] = Complex32::new(ref_samples[i], 0.0);
    }
    fft.process(&mut ref_buf);
    let ref_fft_pre = ref_buf; // reuse

    // spawn streaming ffmpeg child
    let mut child = spawn_ffmpeg_child(stream_path, pan_expr.clone(), Some(target_sr))?;
    let stdout = child.stdout.take().context("no stdout from streaming ffmpeg")?;
    let mut reader = tokio::io::BufReader::new(stdout);

    // prepare reuse buffers
    let overlap_len = if m >= 1 { m - 1 } else { 0 };
    let mut overlap: Vec<f32> = vec![0.0f32; overlap_len];
    let mut local_buf: Vec<u8> = vec![0u8; block_size_b * 4];
    let mut in_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];

    let mut maxv = f32::NEG_INFINITY;
    let mut max_idx_samples: usize = 0usize;
    let mut global_pos: usize = 0usize;

    let _pool = Arc::new(Semaphore::new(pool_capacity));

    loop {
        let nread = reader.read(&mut local_buf).await?;
        if nread == 0 {
            break;
        }
        let samples_read = nread / 4;

        // zero in_buf
        for v in in_buf.iter_mut() {
            *v = Complex32::new(0.0, 0.0);
        }

        // copy overlap
        for (i, &v) in overlap.iter().enumerate() {
            in_buf[i] = Complex32::new(v, 0.0);
        }
        // copy block samples
        for i in 0..samples_read {
            let base = i * 4;
            let bytes = [local_buf[base], local_buf[base + 1], local_buf[base + 2], local_buf[base + 3]];
            let samp = f32::from_le_bytes(bytes);
            let idx = overlap.len() + i;
            if idx < in_buf.len() {
                in_buf[idx] = Complex32::new(samp, 0.0);
            }
        }

        // FFT input (in_buf length n)
        fft.process(&mut in_buf);

        // multiply by ref_fft_pre
        for i in 0..n {
            let a = in_buf[i];
            let b = ref_fft_pre[i];
            in_buf[i] = a * b;
        }

        // inverse
        ifft.process(&mut in_buf);

        // output region start = m-1, length = samples_read (may be < block_size_b)
        let start_idx = m.saturating_sub(1);
        for i in 0..samples_read {
            let idx = start_idx + i;
            if idx >= in_buf.len() {
                break;
            }
            let val = in_buf[idx].re / (n as f32); // rustfft inverse not normalized
            let mag = val.abs();
            if mag > maxv {
                maxv = mag;
                max_idx_samples = global_pos + i;
            }
        }

        // update overlap: take last (m-1) samples from (previous overlap + current block)
        let mut tail_source: Vec<f32> = Vec::with_capacity(overlap_len + samples_read);
        tail_source.extend_from_slice(&overlap);
        // append block samples
        for i in 0..samples_read {
            let base = i * 4;
            let bytes = [local_buf[base], local_buf[base + 1], local_buf[base + 2], local_buf[base + 3]];
            tail_source.push(f32::from_le_bytes(bytes));
        }
        if tail_source.len() >= overlap_len {
            let start = tail_source.len().saturating_sub(overlap_len);
            overlap.clear();
            overlap.extend_from_slice(&tail_source[start..]);
        } else {
            let mut new_overlap = vec![0.0f32; overlap_len];
            let pad = new_overlap.len().saturating_sub(tail_source.len());
            for i in 0..tail_source.len() {
                new_overlap[pad + i] = tail_source[i];
            }
            overlap = new_overlap;
        }

        global_pos += samples_read;
    }

    let _ = child.wait().await;

    let file_to_cut = if ref_path == p1 {
        in2.to_string()
    } else {
        in1.to_string()
    };

    let offset_seconds = (max_idx_samples as f64) / (target_sr as f64);

    Ok(CorrelationResult { file: file_to_cut, offset_seconds })
}

/// Wrapper compatible with your main.rs
pub async fn second_correlation_async(in1: &str, in2: &str, pool_capacity: usize) -> Result<CorrelationResult> {
    second_correlation_streaming(in1, in2, pool_capacity).await
}