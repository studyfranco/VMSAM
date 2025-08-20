use anyhow::{Context, Result};
use num_complex::Complex32;
use rustfft::FftPlanner;
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::Semaphore;

#[derive(Debug, Serialize)]
pub struct CorrelationResult {
    pub file: String,
    pub offset_seconds: f64,
}

/// Tuning constants
const USABLE_PERCENT: usize = 85; // use 85% of detected available memory
const MIN_N_CAP: usize = 1 << 14; // min FFT size (16k)
const ABS_MAX_N_CAP: usize = 1 << 26; // hard cap for FFT size (~67M)
const SAFETY_BYTES_PER_ELEMENT: usize = 20; // bytes per FFT "element" estimation (Complex32 * 2 + headroom)

/// Utility: probe sample rate & duration via ffprobe
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
    Ok((sr.unwrap_or(48000), dur.unwrap_or(0.0)))
}

/// Detect cgroup memory limit (v2 or v1) if present. Returns Some(limit_bytes) or None if no concrete limit.
fn detect_cgroup_limit_bytes() -> Option<usize> {
    // cgroup v2 common file
    let v2_path = Path::new("/sys/fs/cgroup/memory.max");
    if v2_path.exists() {
        if let Ok(s) = fs::read_to_string(v2_path) {
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

    // parse /proc/self/cgroup for v2 path or v1 memory controller
    let cgp = fs::read_to_string("/proc/self/cgroup").ok()?;
    for line in cgp.lines() {
        // v2: "0::/some/path"
        if line.starts_with("0::") {
            if let Some(path_part) = line.splitn(3, ':').nth(2) {
                let path_trim = if path_part.starts_with('/') { &path_part[1..] } else { path_part };
                let candidate = Path::new("/sys/fs/cgroup").join(path_trim).join("memory.max");
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
            }
        }
    }

    // try v1: find controller containing "memory"
    for line in cgp.lines() {
        let parts: Vec<&str> = line.splitn(3, ':').collect();
        if parts.len() == 3 {
            let controllers = parts[1];
            let cpath = parts[2];
            if controllers.split(',').any(|c| c == "memory") {
                let cpath_trim = if cpath.starts_with('/') { &cpath[1..] } else { cpath };
                let candidate = Path::new("/sys/fs/cgroup/memory").join(cpath_trim).join("memory.limit_in_bytes");
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
            }
        }
    }

    None
}

/// Read MemAvailable from /proc/meminfo. Fallback to MemTotal or 512MB if unavailable.
fn read_mem_available_bytes() -> usize {
    const DEFAULT: usize = 512 * 1024 * 1024;
    if let Ok(s) = fs::read_to_string("/proc/meminfo") {
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
    }
    DEFAULT
}

/// Compute available memory (considers cgroup limit if present).
fn detect_available_memory_bytes() -> usize {
    let meminfo = read_mem_available_bytes();
    if let Some(cg) = detect_cgroup_limit_bytes() {
        std::cmp::min(cg, meminfo)
    } else {
        meminfo
    }
}

/// Compute usable FFT cap (power of two) based on available memory and USABLE_PERCENT.
fn compute_max_n_cap() -> usize {
    let avail = detect_available_memory_bytes();
    let usable = avail.saturating_mul(USABLE_PERCENT) / 100;
    if usable == 0 {
        return MIN_N_CAP;
    }
    let mut n = usable / SAFETY_BYTES_PER_ELEMENT;
    if n < MIN_N_CAP {
        n = MIN_N_CAP;
    }
    if n > ABS_MAX_N_CAP {
        n = ABS_MAX_N_CAP;
    }
    // round down to power of two
    let p = next_pow2_floor(n);
    if p < MIN_N_CAP { MIN_N_CAP } else { p }
}

/// next pow2 >= n
fn next_pow2(mut n: usize) -> usize {
    if n == 0 { return 1; }
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
    if n == 0 { return 1; }
    if n.is_power_of_two() { return n; }
    let mut p = 1usize;
    while p <= n { p <<= 1; }
    p >> 1
}

/// Read entire audio via ffmpeg into Vec<f32> (mono, target sample rate).
/// Uses ffmpeg to mix to mono and resample if needed. Returns (sr, samples).
async fn read_full_pcm_f32(path: &Path, target_sr: u32) -> Result<(u32, Vec<f32>)> {
    let mut args = vec![
        "-vn".to_string(),
        "-nostdin".to_string(),
        "-i".to_string(),
        path.to_string_lossy().into_owned(),
        "-ac".to_string(),
        "1".to_string(),                          // downmix to mono
        "-ar".to_string(),
        target_sr.to_string(),                    // resample
        "-af".to_string(),
        "loudnorm=i=-23.0:lra=7.0:tp=-2.0:linear=true:print_format=json".to_string(), // loudness normalization
        "-f".to_string(),
        "f32le".to_string(),
        "-acodec".to_string(),
        "pcm_f32le".to_string(),
        "-".to_string(),
    ];

    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("spawn ffmpeg for full read")?;

    let stdout = child.stdout.take().context("no stdout from ffmpeg")?;
    let mut reader = tokio::io::BufReader::new(stdout);

    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).await.context("read ffmpeg stdout")?;
    let _ = child.wait().await;

    let mut samples = Vec::with_capacity(buf.len() / 4);
    for chunk in buf.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        samples.push(f32::from_le_bytes(bytes));
    }
    Ok((target_sr, samples))
}

/// Correlate fully in-memory using one big FFT (s1 and s2 are mono f32 samples).
fn correlate_full(s1: &[f32], s2: &[f32]) -> Result<(usize, usize)> {
    // compute needed padsize
    let ls1 = s1.len();
    let ls2 = s2.len();
    if ls1 == 0 || ls2 == 0 {
        anyhow::bail!("empty input for correlation");
    }
    let needed = ls1 + ls2 - 1;
    let mut n = next_pow2(needed);
    // create planner and buffers
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut a: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    let mut b: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];

    for i in 0..ls1 { a[i] = Complex32::new(s1[i], 0.0); }
    for i in 0..ls2 { b[i] = Complex32::new(s2[i], 0.0); }

    fft.process(&mut a);
    fft.process(&mut b);

    for i in 0..n {
        let conj_b = Complex32::new(b[i].re, -b[i].im);
        a[i] *= conj_b;
    }

    ifft.process(&mut a);

    // find max magnitude
    let mut xmax = 0usize;
    let mut maxv = f32::NEG_INFINITY;
    for (i, v) in a.iter().enumerate() {
        let mag = (v.re * v.re + v.im * v.im).sqrt();
        if mag > maxv {
            maxv = mag;
            xmax = i;
        }
    }

    Ok((n, xmax))
}

/// Overlap-save correlation streaming: ref_samples is the (reversed) reference already in memory,
/// stream_path is the path to feed ffmpeg which outputs pcm_f32le mono resampled to target_sr.
async fn correlate_overlap_save(
    ref_samples_rev: &[f32],
    stream_path: &Path,
    target_sr: u32,
    max_n_cap: usize,
    pool_capacity: usize,
) -> Result<(usize, usize)> {
    let m = ref_samples_rev.len();
    if m == 0 { anyhow::bail!("empty reference"); }
    // choose n: largest power of two <= max_n_cap but >= m
    let mut n = next_pow2_floor(max_n_cap);
    if n < m {
        n = next_pow2(m);
        if n > max_n_cap {
            anyhow::bail!("reference too large for overlap-save with cap (m={}, cap={})", m, max_n_cap);
        }
    }

    // block size B = N - m + 1
    let block_b = n.saturating_sub(m).saturating_add(1);
    if block_b == 0 { anyhow::bail!("computed block size is zero"); }

    // prepare planner and precompute ref_fft at n
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut ref_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..m.min(n) {
        ref_buf[i] = Complex32::new(ref_samples_rev[i], 0.0);
    }
    fft.process(&mut ref_buf);
    let ref_fft = ref_buf;

    // spawn ffmpeg for stream_path
    let mut args = vec![
        "-vn".to_string(),
        "-nostdin".to_string(),
        "-i".to_string(),
        stream_path.to_string_lossy().into_owned(),
        "-ac".to_string(),
        "1".to_string(),                          // downmix to mono
        "-ar".to_string(),
        target_sr.to_string(),                    // resample
        "-af".to_string(),
        "loudnorm=i=-23.0:lra=7.0:tp=-2.0:linear=true:print_format=json".to_string(), // loudness normalization
        "-f".to_string(),
        "f32le".to_string(),
        "-acodec".to_string(),
        "pcm_f32le".to_string(),
        "-".to_string(),
    ];
    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("spawn ffmpeg for streaming")?;

    let stdout = child.stdout.take().context("no stdout from ffmpeg")?;
    let mut reader = tokio::io::BufReader::new(stdout);

    // buffers reused
    let overlap_len = m.saturating_sub(1);
    let mut overlap: Vec<f32> = vec![0.0f32; overlap_len];
    let mut local_bytes: Vec<u8> = vec![0u8; block_b.saturating_mul(4)];
    let mut in_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];

    let mut maxv = f32::NEG_INFINITY;
    let mut max_idx_samples: usize = 0usize;
    let mut global_pos: usize = 0usize;

    let _pool = Arc::new(Semaphore::new(pool_capacity));

    loop {
        let nread = reader.read(&mut local_bytes).await?;
        if nread == 0 { break; }
        let samples_read = nread / 4;

        // zero in_buf
        for v in in_buf.iter_mut() { *v = Complex32::new(0.0, 0.0); }

        // copy overlap
        for (i, &v) in overlap.iter().enumerate() {
            in_buf[i] = Complex32::new(v, 0.0);
        }
        // copy block samples
        for i in 0..samples_read {
            let base = i * 4;
            let bytes = [local_bytes[base], local_bytes[base+1], local_bytes[base+2], local_bytes[base+3]];
            let samp = f32::from_le_bytes(bytes);
            let idx = overlap.len() + i;
            if idx < in_buf.len() {
                in_buf[idx] = Complex32::new(samp, 0.0);
            }
        }

        // fft input
        fft.process(&mut in_buf);
        // multiply with ref_fft
        for i in 0..n {
            in_buf[i] = in_buf[i] * ref_fft[i];
        }
        // inverse
        ifft.process(&mut in_buf);

        // output region start = m-1
        let start_idx = m.saturating_sub(1);
        for i in 0..samples_read {
            let idx = start_idx + i;
            if idx >= in_buf.len() { break; }
            let val = in_buf[idx].re / (n as f32);
            let mag = val.abs();
            if mag > maxv {
                maxv = mag;
                max_idx_samples = global_pos + i;
            }
        }

        // update overlap: keep last overlap_len samples of (overlap + block)
        let mut tail: Vec<f32> = Vec::with_capacity(overlap_len + samples_read);
        tail.extend_from_slice(&overlap);
        for i in 0..samples_read {
            let base = i * 4;
            let bytes = [local_bytes[base], local_bytes[base+1], local_bytes[base+2], local_bytes[base+3]];
            tail.push(f32::from_le_bytes(bytes));
        }
        if tail.len() >= overlap_len {
            let start = tail.len().saturating_sub(overlap_len);
            overlap.clear();
            overlap.extend_from_slice(&tail[start..]);
        } else {
            let mut new_ov = vec![0.0f32; overlap_len];
            let pad = new_ov.len().saturating_sub(tail.len());
            for i in 0..tail.len() {
                new_ov[pad + i] = tail[i];
            }
            overlap = new_ov;
        }

        global_pos += samples_read;
    }

    let _ = child.wait().await;
    Ok((n, max_idx_samples))
}

/// Top-level function exposed to main.rs: choose fast full-FFT if memory allows, else overlap-save streaming.
pub async fn second_correlation_async(in1: &str, in2: &str, pool_capacity: usize) -> Result<CorrelationResult> {
    // probe both files
    let p1 = Path::new(in1);
    let p2 = Path::new(in2);
    let (sr1, dur1) = probe_samplerate_duration(p1).await?;
    let (sr2, dur2) = probe_samplerate_duration(p2).await?;

    // choose common target sr
    let target_sr = std::cmp::min(sr1, sr2);

    // estimate samples
    let est1 = (sr1 as f64 * dur1).round() as usize;
    let est2 = (sr2 as f64 * dur2).round() as usize;

    // compute memory-based cap
    let max_n_cap = compute_max_n_cap();

    // Estimate needed N for full FFT
    let needed_full = est1.saturating_add(est2).saturating_sub(1);
    let n_full = next_pow2(needed_full);

    // compute estimated bytes for full in-memory FFT: conservative
    let bytes_needed = (n_full as usize)
        .saturating_mul(SAFETY_BYTES_PER_ELEMENT)
        .saturating_mul(2) / 2; // keep conservative factor ~1x

    let avail = detect_available_memory_bytes();
    let usable = avail.saturating_mul(USABLE_PERCENT) / 100;

    if bytes_needed <= usable && n_full <= ABS_MAX_N_CAP {
        // fast path: load both files fully and run correlate_full
        let (_sr_a, a) = read_full_pcm_f32(p1, target_sr).await?;
        let (_sr_b, b) = read_full_pcm_f32(p2, target_sr).await?;
        let (padsize, xmax) = correlate_full(&a, &b)?;
        let fs_f = target_sr as f64;
        let (file_cut, offset_seconds) = if xmax > padsize / 2 {
            (in2.to_string(), (padsize - xmax) as f64 / fs_f)
        } else {
            (in1.to_string(), (xmax) as f64 / fs_f)
        };
        return Ok(CorrelationResult { file: file_cut, offset_seconds });
    }

    // otherwise streaming: choose shorter file as reference to load fully
    let (ref_path, stream_path, ref_est) = if est1 <= est2 {
        (p1, p2, est1)
    } else {
        (p2, p1, est2)
    };

    // read reference fully
    let (_sr_ref, mut ref_samples) = read_full_pcm_f32(ref_path, target_sr).await?;
    if ref_samples.is_empty() {
        anyhow::bail!("reference empty after read");
    }
    // reverse reference for convolution
    ref_samples.reverse();
    let m = ref_samples.len();
    if m > max_n_cap {
        anyhow::bail!(
            "reference is too large for streaming FFT given current memory cap (m={}, cap={}). Consider increasing container memory.",
            m,
            max_n_cap
        );
    }

    // pick n: try to make blocks big (so fewer FFTs) but <= max_n_cap
    // target block size B_target = something proportional to usable memory. We'll set N = min(max_n_cap, next_pow2(m + B_desired - 1))
    // choose B_desired = min( est_stream, max( BLOCK ~ m or something ) ). For simplicity pick B_desired = max( BLOCK = m * 2, 1<<16 ) but bounded by memory.
    let desired_b = std::cmp::max(m.saturating_mul(2), 1 << 16);
    let mut n_try = next_pow2(m.saturating_add(desired_b).saturating_sub(1));
    if n_try > max_n_cap {
        n_try = next_pow2_floor(max_n_cap);
        if n_try < m {
            n_try = next_pow2(m);
            if n_try > max_n_cap {
                anyhow::bail!("cannot find FFT size >= m within cap");
            }
        }
    }
    let n = n_try;

    // run overlap-save with chosen n
    let (padsize, xmax_samples) = correlate_overlap_save(&ref_samples, stream_path, target_sr, n, pool_capacity).await?;

    // interpret result
    let fs_f = target_sr as f64;
    // here overlap-save returns global sample index xmax_samples within stream where correlation peak occurred.
    // If ref_path == p1 then we want to cut in2; else cut in1.
    let file_cut = if ref_path == p1 { in2.to_string() } else { in1.to_string() };
    let offset_seconds = (xmax_samples as f64) / fs_f;

    Ok(CorrelationResult { file: file_cut, offset_seconds })
}