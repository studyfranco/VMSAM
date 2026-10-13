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

/// The file to cut at its start and by how many seconds to bring the two files in sync.
#[derive(Debug, Serialize)]
pub struct CorrelationResult {
    pub file: String,
    pub offset_seconds: f64,
}

// Tuning constants.
const USABLE_PERCENT: usize = 80; // percent of available memory the FFT may use
const MIN_N_CAP: usize = 1 << 16; // minimum FFT size
const ABS_MAX_N_CAP: usize = 1 << 28; // hard cap on FFT size
const SAFETY_BYTES_PER_ELEMENT: usize = 18; // estimated bytes per FFT element (two Complex32 + headroom)

/// Probe the first audio stream's sample rate and duration (seconds) with ffprobe.
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
    Ok((sr.unwrap_or(44100), dur.unwrap_or(0.0)))
}

/// Return the cgroup (v2 or v1) memory limit in bytes, or None when there is no concrete limit.
fn detect_cgroup_limit_bytes() -> Option<usize> {
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

    // Resolve this process's cgroup path from /proc/self/cgroup.
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

/// Return MemAvailable from /proc/meminfo, falling back to MemTotal, then 512 MB.
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

/// Return the available memory in bytes, capped by the cgroup limit if any.
fn detect_available_memory_bytes() -> usize {
    let meminfo = read_mem_available_bytes();
    if let Some(cg) = detect_cgroup_limit_bytes() {
        std::cmp::min(cg, meminfo)
    } else {
        meminfo
    }
}

/// Return the largest FFT size (power of two) that fits in USABLE_PERCENT of available memory.
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
    let p = next_pow2_floor(n);
    if p < MIN_N_CAP { MIN_N_CAP } else { p }
}

/// Smallest power of two >= n.
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

/// Largest power of two <= n.
fn next_pow2_floor(n: usize) -> usize {
    if n == 0 { return 1; }
    if n.is_power_of_two() { return n; }
    let mut p = 1usize;
    while p <= n { p <<= 1; }
    p >> 1
}

/// Decode a whole file with ffmpeg to mono f32 samples at `target_sr`; returns (sr, samples).
async fn read_full_pcm_f32(path: &Path, target_sr: u32) -> Result<(u32, Vec<f32>)> {
    let args = vec![
        "-probesize".to_string(),
        "1000M".to_string(),
        "-threads".to_string(),
        "3".to_string(),
        "-vn".to_string(),
        "-nostdin".to_string(),
        "-i".to_string(),
        path.to_string_lossy().into_owned(),
        "-ac".to_string(),
        "1".to_string(),                          // downmix to mono
        "-ar".to_string(),
        target_sr.to_string(),                    // resample
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

/// Cross-correlate two signals in memory with one FFT; returns (padsize, peak index).
fn correlate_full(s1: &[f32], s2: &[f32]) -> Result<(usize, usize)> {
    let ls1 = s1.len();
    let ls2 = s2.len();
    if ls1 == 0 || ls2 == 0 {
        anyhow::bail!("empty input for correlation");
    }
    let needed = ls1 + ls2 - 1;
    let n = next_pow2(needed);
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

/// Cross-correlate a reversed in-memory reference against a file streamed through ffmpeg,
/// using overlap-save with FFT size `n`; returns (n, global sample index of the peak).
async fn correlate_overlap_save(
    ref_samples_rev: &[f32],
    stream_path: &Path,
    target_sr: u32,
    max_n_cap: usize,
    pool_capacity: usize,
) -> Result<(usize, usize)> {
    let m = ref_samples_rev.len();
    if m == 0 { anyhow::bail!("empty reference"); }
    let mut n = next_pow2_floor(max_n_cap);
    if n < m {
        n = next_pow2(m);
        if n > max_n_cap {
            anyhow::bail!("reference too large for overlap-save with cap (m={}, cap={})", m, max_n_cap);
        }
    }

    // Overlap-save block size: B = N - m + 1.
    let block_b = n.saturating_sub(m).saturating_add(1);
    if block_b == 0 { anyhow::bail!("computed block size is zero"); }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut ref_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..m.min(n) {
        ref_buf[i] = Complex32::new(ref_samples_rev[i], 0.0);
    }
    fft.process(&mut ref_buf);
    let ref_fft = ref_buf;

    let args = vec![
        "-threads".to_string(),
        "3".to_string(),
        "-vn".to_string(),
        "-nostdin".to_string(),
        "-i".to_string(),
        stream_path.to_string_lossy().into_owned(),
        "-ac".to_string(),
        "1".to_string(),                          // downmix to mono
        "-ar".to_string(),
        target_sr.to_string(),                    // resample
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

        for v in in_buf.iter_mut() { *v = Complex32::new(0.0, 0.0); }

        for (i, &v) in overlap.iter().enumerate() {
            in_buf[i] = Complex32::new(v, 0.0);
        }
        for i in 0..samples_read {
            let base = i * 4;
            let bytes = [local_bytes[base], local_bytes[base+1], local_bytes[base+2], local_bytes[base+3]];
            let samp = f32::from_le_bytes(bytes);
            let idx = overlap.len() + i;
            if idx < in_buf.len() {
                in_buf[idx] = Complex32::new(samp, 0.0);
            }
        }

        fft.process(&mut in_buf);
        for i in 0..n {
            in_buf[i] = in_buf[i] * ref_fft[i];
        }
        ifft.process(&mut in_buf);

        // Valid (non-aliased) output starts at index m-1.
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

        // Keep the last m-1 samples of (overlap + block) for the next block.
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

/// Find the offset aligning two audio files and which file to cut.
///
/// Uses one in-memory FFT when memory allows, else overlap-save streaming
/// with the shorter file as the reference.
pub async fn second_correlation_async(in1: &str, in2: &str, pool_capacity: usize) -> Result<CorrelationResult> {
    let p1 = Path::new(in1);
    let p2 = Path::new(in2);
    let (sr1, dur1) = probe_samplerate_duration(p1).await?;
    let (sr2, dur2) = probe_samplerate_duration(p2).await?;

    let target_sr = std::cmp::min(sr1, sr2);

    let est1 = (sr1 as f64 * dur1).round() as usize;
    let est2 = (sr2 as f64 * dur2).round() as usize;

    let max_n_cap = compute_max_n_cap();

    let needed_full = est1.saturating_add(est2).saturating_sub(1);
    let n_full = next_pow2(needed_full);

    let bytes_needed = (n_full as usize)
        .saturating_mul(SAFETY_BYTES_PER_ELEMENT)
        .saturating_mul(2) / 2;

    let avail = detect_available_memory_bytes();
    let usable = avail.saturating_mul(USABLE_PERCENT) / 100;

    if bytes_needed <= usable && n_full <= ABS_MAX_N_CAP {
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

    // Streaming: the shorter file is the in-memory reference.
    let (ref_path, stream_path, _ref_est) = if est1 <= est2 {
        (p1, p2, est1)
    } else {
        (p2, p1, est2)
    };

    let (_sr_ref, mut ref_samples) = read_full_pcm_f32(ref_path, target_sr).await?;
    if ref_samples.is_empty() {
        anyhow::bail!("reference empty after read");
    }
    // Reversed so that convolution computes correlation.
    ref_samples.reverse();
    let m = ref_samples.len();
    if m > max_n_cap {
        anyhow::bail!(
            "reference is too large for streaming FFT given current memory cap (m={}, cap={}). Consider increasing container memory.",
            m,
            max_n_cap
        );
    }

    // FFT size: room for blocks of max(2m, 64k) samples (fewer FFTs), bounded by the memory cap.
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

    let (_padsize, xmax_samples) = correlate_overlap_save(&ref_samples, stream_path, target_sr, n, pool_capacity).await?;

    let fs_f = target_sr as f64;
    // The peak index is a sample position in the streamed file, which is the one to cut.
    let file_cut = if ref_path == p1 { in2.to_string() } else { in1.to_string() };
    let offset_seconds = (xmax_samples as f64) / fs_f;

    Ok(CorrelationResult { file: file_cut, offset_seconds })
}