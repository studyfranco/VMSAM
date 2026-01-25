use anyhow::{Context, Result};
use num_complex::Complex32;
use rayon::prelude::*;
use rustfft::FftPlanner;
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

#[derive(Debug, Serialize)]
pub struct CorrelationResult {
    pub file: String,
    pub offset_seconds: f64,
}

/// Tuning constants
const USABLE_PERCENT: usize = 70; // Use 70% of detected available memory (safe balance)
const MIN_N_CAP: usize = 1 << 16; // min FFT size (64k)
const ABS_MAX_N_CAP: usize = 1 << 28; // hard cap for FFT size (~268M)
const SAFETY_BYTES_PER_ELEMENT: usize = 18; // bytes per FFT "element" estimation
const CHUNK_SIZE_SAMPLES: usize = 262144; // 2^18 samples - fits in L3 cache (~2MB per buffer)

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

/// Detect cgroup memory limit (v2 or v1) if present.
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
                let path_trim = if path_part.starts_with('/') {
                    &path_part[1..]
                } else {
                    path_part
                };
                let candidate = Path::new("/sys/fs/cgroup")
                    .join(path_trim)
                    .join("memory.max");
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
                let cpath_trim = if cpath.starts_with('/') {
                    &cpath[1..]
                } else {
                    cpath
                };
                let candidate = Path::new("/sys/fs/cgroup/memory")
                    .join(cpath_trim)
                    .join("memory.limit_in_bytes");
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

/// Read MemAvailable from /proc/meminfo.
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

/// Check for environment variable override (container safety)
fn get_env_memory_limit() -> Option<usize> {
    std::env::var("VMSAM_MEMORY_LIMIT_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|mb| mb * 1024 * 1024)
}

/// Compute available memory (considers cgroup limit and env override).
fn detect_available_memory_bytes() -> usize {
    let meminfo = read_mem_available_bytes();
    let mut limit = meminfo;

    if let Some(cg) = detect_cgroup_limit_bytes() {
        limit = std::cmp::min(limit, cg);
    }

    if let Some(env_limit) = get_env_memory_limit() {
        limit = std::cmp::min(limit, env_limit);
    }

    limit
}

/// Compute usable FFT cap based on available memory.
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
    if p < MIN_N_CAP {
        MIN_N_CAP
    } else {
        p
    }
}

/// next pow2 >= n
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

/// Read entire audio via ffmpeg into Vec<i16> (mono, target sample rate).
/// Uses i16 for memory efficiency - converts to Complex32 on-the-fly during FFT.
async fn read_full_pcm_i16(path: &Path, target_sr: u32) -> Result<(u32, Vec<i16>)> {
    let args = vec![
        "-threads".to_string(),
        "3".to_string(),
        "-vn".to_string(),
        "-nostdin".to_string(),
        "-i".to_string(),
        path.to_string_lossy().into_owned(),
        "-ac".to_string(),
        "1".to_string(),
        "-ar".to_string(),
        target_sr.to_string(),
        "-f".to_string(),
        "s16le".to_string(),
        "-acodec".to_string(),
        "pcm_s16le".to_string(),
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
    reader
        .read_to_end(&mut buf)
        .await
        .context("read ffmpeg stdout")?;
    let _ = child.wait().await;

    // Convert bytes to i16
    let mut samples = Vec::with_capacity(buf.len() / 2);
    for chunk in buf.chunks_exact(2) {
        let bytes = [chunk[0], chunk[1]];
        samples.push(i16::from_le_bytes(bytes));
    }
    Ok((target_sr, samples))
}

/// Convert i16 samples to Complex32, normalizing to [-1, 1] range
#[inline]
fn i16_to_complex(sample: i16) -> Complex32 {
    Complex32::new(sample as f32 / 32768.0, 0.0)
}

/// Cache-friendly parallel FFT correlation using chunks that fit in L3 cache.
/// CRITICAL: The smaller vector MUST be the reference (kernel) to enable chunking.
fn correlate_parallel(reference: &[i16], signal: &[i16]) -> Result<(usize, usize)> {
    let ref_len = reference.len();
    let sig_len = signal.len();

    if ref_len == 0 || sig_len == 0 {
        anyhow::bail!("empty input for correlation");
    }

    // Total correlation length
    let corr_len = ref_len + sig_len - 1;
    let fft_size = next_pow2(corr_len);

    // Check if we can do a single FFT (small signals)
    if sig_len <= CHUNK_SIZE_SAMPLES * 2 {
        return correlate_single_fft(reference, signal);
    }

    // Use overlap-save chunking for large signals
    // Each chunk is CHUNK_SIZE_SAMPLES, with overlap of (ref_len - 1)
    let overlap = ref_len.saturating_sub(1);
    let step = CHUNK_SIZE_SAMPLES.saturating_sub(overlap);
    if step == 0 {
        // Reference too large for chunking, fall back to single FFT
        return correlate_single_fft(reference, signal);
    }

    // Chunk FFT size: enough to hold chunk + reference for linear conv
    let chunk_fft_size = next_pow2(CHUNK_SIZE_SAMPLES + ref_len - 1);

    // Precompute reference FFT at chunk_fft_size (zero-padded)
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(chunk_fft_size);

    let mut ref_padded: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); chunk_fft_size];
    for (i, &s) in reference.iter().enumerate() {
        ref_padded[i] = i16_to_complex(s);
    }
    fft.process(&mut ref_padded);
    let ref_fft = Arc::new(ref_padded);

    // Calculate number of chunks
    let num_chunks = (sig_len + step - 1) / step;

    // Process chunks in parallel using rayon
    let chunk_results: Vec<(f32, usize)> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * step;
            let end = std::cmp::min(start + CHUNK_SIZE_SAMPLES, sig_len);
            let chunk_len = end - start;

            // Create local planner (planners are not thread-safe for sharing)
            let mut local_planner = FftPlanner::<f32>::new();
            let local_fft = local_planner.plan_fft_forward(chunk_fft_size);
            let local_ifft = local_planner.plan_fft_inverse(chunk_fft_size);

            // Prepare signal chunk (zero-padded)
            let mut sig_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); chunk_fft_size];
            for (i, &s) in signal[start..end].iter().enumerate() {
                sig_buf[i] = i16_to_complex(s);
            }

            // FFT of signal chunk
            local_fft.process(&mut sig_buf);

            // Multiply with conjugate of reference FFT
            for (i, ref_val) in ref_fft.iter().enumerate() {
                let conj = Complex32::new(ref_val.re, -ref_val.im);
                sig_buf[i] *= conj;
            }

            // IFFT
            local_ifft.process(&mut sig_buf);

            // Find max in valid output region
            // Valid output indices: [ref_len-1, ref_len-1 + chunk_len - 1]
            // But we only trust the non-overlapping part for non-first chunks
            let valid_start = if chunk_idx == 0 { 0 } else { overlap };
            let valid_end = std::cmp::min(chunk_len + ref_len - 1, chunk_fft_size);

            let mut local_max = f32::NEG_INFINITY;
            let mut local_max_idx = 0usize;

            for i in valid_start..valid_end {
                let mag = sig_buf[i].norm();
                if mag > local_max {
                    local_max = mag;
                    // Global index: start + (i - (ref_len - 1)) but adjusted for output
                    local_max_idx = start + i;
                }
            }

            (local_max, local_max_idx)
        })
        .collect();

    // Find global maximum
    let mut global_max = f32::NEG_INFINITY;
    let mut global_idx = 0usize;
    for (max_val, idx) in chunk_results {
        if max_val > global_max {
            global_max = max_val;
            global_idx = idx;
        }
    }

    Ok((fft_size, global_idx))
}

/// Single FFT correlation for smaller signals (fits in memory)
fn correlate_single_fft(reference: &[i16], signal: &[i16]) -> Result<(usize, usize)> {
    let ref_len = reference.len();
    let sig_len = signal.len();

    if ref_len == 0 || sig_len == 0 {
        anyhow::bail!("empty input for correlation");
    }

    let needed = ref_len + sig_len - 1;
    let n = next_pow2(needed);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Allocate and fill buffers (i16 -> Complex32 on-the-fly)
    let mut a: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    let mut b: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];

    // Parallel conversion for large arrays
    if ref_len > 10000 {
        a.par_iter_mut()
            .enumerate()
            .take(ref_len)
            .for_each(|(i, v)| *v = i16_to_complex(reference[i]));
        b.par_iter_mut()
            .enumerate()
            .take(sig_len)
            .for_each(|(i, v)| *v = i16_to_complex(signal[i]));
    } else {
        for (i, &s) in reference.iter().enumerate() {
            a[i] = i16_to_complex(s);
        }
        for (i, &s) in signal.iter().enumerate() {
            b[i] = i16_to_complex(s);
        }
    }

    fft.process(&mut a);
    fft.process(&mut b);

    // Multiply a by conjugate of b (parallel for large n)
    if n > 100000 {
        a.par_iter_mut().zip(b.par_iter()).for_each(|(av, bv)| {
            let conj_b = Complex32::new(bv.re, -bv.im);
            *av *= conj_b;
        });
    } else {
        for i in 0..n {
            let conj_b = Complex32::new(b[i].re, -b[i].im);
            a[i] *= conj_b;
        }
    }

    ifft.process(&mut a);

    // Find max magnitude (parallel for large n)
    let (xmax, _) = if n > 100000 {
        a.par_iter()
            .enumerate()
            .map(|(i, v)| (i, v.norm()))
            .reduce(
                || (0, f32::NEG_INFINITY),
                |(i1, v1), (i2, v2)| if v1 > v2 { (i1, v1) } else { (i2, v2) },
            )
    } else {
        let mut xmax = 0usize;
        let mut maxv = f32::NEG_INFINITY;
        for (i, v) in a.iter().enumerate() {
            let mag = v.norm();
            if mag > maxv {
                maxv = mag;
                xmax = i;
            }
        }
        (xmax, maxv)
    };

    Ok((n, xmax))
}

/// Read entire audio via ffmpeg into Vec<f32> (for streaming fallback)
async fn read_full_pcm_f32(path: &Path, target_sr: u32) -> Result<(u32, Vec<f32>)> {
    let args = vec![
        "-threads".to_string(),
        "3".to_string(),
        "-vn".to_string(),
        "-nostdin".to_string(),
        "-i".to_string(),
        path.to_string_lossy().into_owned(),
        "-ac".to_string(),
        "1".to_string(),
        "-ar".to_string(),
        target_sr.to_string(),
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
    reader
        .read_to_end(&mut buf)
        .await
        .context("read ffmpeg stdout")?;
    let _ = child.wait().await;

    let mut samples = Vec::with_capacity(buf.len() / 4);
    for chunk in buf.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        samples.push(f32::from_le_bytes(bytes));
    }
    Ok((target_sr, samples))
}

/// Overlap-save correlation streaming (fallback for OOM scenarios)
async fn correlate_overlap_save_streaming(
    ref_samples_rev: &[f32],
    stream_path: &Path,
    target_sr: u32,
    max_n_cap: usize,
) -> Result<(usize, usize)> {
    let m = ref_samples_rev.len();
    if m == 0 {
        anyhow::bail!("empty reference");
    }

    let mut n = next_pow2_floor(max_n_cap);
    if n < m {
        n = next_pow2(m);
        if n > max_n_cap {
            anyhow::bail!(
                "reference too large for overlap-save with cap (m={}, cap={})",
                m,
                max_n_cap
            );
        }
    }

    let block_b = n.saturating_sub(m).saturating_add(1);
    if block_b == 0 {
        anyhow::bail!("computed block size is zero");
    }

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
        "1".to_string(),
        "-ar".to_string(),
        target_sr.to_string(),
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
    let mut max_idx_samples: usize = 0;
    let mut global_pos: usize = 0;

    loop {
        let nread = reader.read(&mut local_bytes).await?;
        if nread == 0 {
            break;
        }
        let samples_read = nread / 4;

        for v in in_buf.iter_mut() {
            *v = Complex32::new(0.0, 0.0);
        }

        for (i, &v) in overlap.iter().enumerate() {
            in_buf[i] = Complex32::new(v, 0.0);
        }

        for i in 0..samples_read {
            let base = i * 4;
            let bytes = [
                local_bytes[base],
                local_bytes[base + 1],
                local_bytes[base + 2],
                local_bytes[base + 3],
            ];
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

        let start_idx = m.saturating_sub(1);
        for i in 0..samples_read {
            let idx = start_idx + i;
            if idx >= in_buf.len() {
                break;
            }
            let val = in_buf[idx].re / (n as f32);
            let mag = val.abs();
            if mag > maxv {
                maxv = mag;
                max_idx_samples = global_pos + i;
            }
        }

        let mut tail: Vec<f32> = Vec::with_capacity(overlap_len + samples_read);
        tail.extend_from_slice(&overlap);
        for i in 0..samples_read {
            let base = i * 4;
            let bytes = [
                local_bytes[base],
                local_bytes[base + 1],
                local_bytes[base + 2],
                local_bytes[base + 3],
            ];
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

/// Top-level function: Parallel I/O + Cache-Friendly Compute pattern
pub async fn second_correlation_async(
    in1: &str,
    in2: &str,
    _pool_capacity: usize, // kept for API compatibility
) -> Result<CorrelationResult> {
    let p1 = Path::new(in1);
    let p2 = Path::new(in2);

    // Probe both files concurrently
    let (probe1, probe2) = tokio::try_join!(
        probe_samplerate_duration(p1),
        probe_samplerate_duration(p2)
    )?;
    let (sr1, dur1) = probe1;
    let (sr2, dur2) = probe2;

    // Choose common target sample rate (lower of the two)
    let target_sr = std::cmp::min(sr1, sr2);

    // Estimate samples
    let est1 = (sr1 as f64 * dur1).round() as usize;
    let est2 = (sr2 as f64 * dur2).round() as usize;

    // Compute memory-based cap
    let max_n_cap = compute_max_n_cap();

    // Estimate needed N for full FFT
    let needed_full = est1.saturating_add(est2).saturating_sub(1);
    let n_full = next_pow2(needed_full);

    // Compute estimated bytes needed
    // i16 storage: 2 bytes/sample, Complex32 during FFT: 8 bytes/element
    let bytes_for_storage = (est1 + est2) * 2; // i16 storage
    let bytes_for_fft = n_full * 8 * 2; // Two Complex32 buffers during FFT
    let bytes_needed = bytes_for_storage + bytes_for_fft;

    let avail = detect_available_memory_bytes();
    let usable = avail.saturating_mul(USABLE_PERCENT) / 100;

    if bytes_needed <= usable && n_full <= ABS_MAX_N_CAP {
        // FAST PATH: Parallel I/O + Cache-Friendly Parallel FFT

        // Read both audio files CONCURRENTLY using try_join!
        let (read1, read2) = tokio::try_join!(
            read_full_pcm_i16(p1, target_sr),
            read_full_pcm_i16(p2, target_sr)
        )?;

        let (_, samples1) = read1;
        let (_, samples2) = read2;

        // CRITICAL: Ensure smaller is reference (kernel) for chunking
        let (reference, signal, ref_is_first) = if samples1.len() <= samples2.len() {
            (&samples1, &samples2, true)
        } else {
            (&samples2, &samples1, false)
        };

        // Run cache-friendly parallel correlation
        let (padsize, xmax) = correlate_parallel(reference, signal)?;

        let fs_f = target_sr as f64;

        // Interpret result: xmax is the lag where reference aligns with signal
        let (file_cut, offset_seconds) = if xmax > padsize / 2 {
            // Negative lag
            let offset = (padsize - xmax) as f64 / fs_f;
            if ref_is_first {
                (in2.to_string(), offset)
            } else {
                (in1.to_string(), offset)
            }
        } else {
            // Positive lag
            let offset = xmax as f64 / fs_f;
            if ref_is_first {
                (in1.to_string(), offset)
            } else {
                (in2.to_string(), offset)
            }
        };

        return Ok(CorrelationResult {
            file: file_cut,
            offset_seconds,
        });
    }

    // SLOW PATH (Fallback): Streaming overlap-save for OOM predictions
    // Choose shorter file as reference
    let (ref_path, stream_path, _ref_est) = if est1 <= est2 {
        (p1, p2, est1)
    } else {
        (p2, p1, est2)
    };

    // Read reference fully
    let (_sr_ref, mut ref_samples) = read_full_pcm_f32(ref_path, target_sr).await?;
    if ref_samples.is_empty() {
        anyhow::bail!("reference empty after read");
    }

    // Reverse reference for convolution
    ref_samples.reverse();
    let m = ref_samples.len();

    if m > max_n_cap {
        anyhow::bail!(
            "reference too large for streaming FFT (m={}, cap={}). Increase container memory.",
            m,
            max_n_cap
        );
    }

    // Run overlap-save streaming (sequential is fine for fallback, stability > speed)
    let (_padsize, xmax_samples) =
        correlate_overlap_save_streaming(&ref_samples, stream_path, target_sr, max_n_cap).await?;

    let fs_f = target_sr as f64;
    let file_cut = if ref_path == p1 {
        in2.to_string()
    } else {
        in1.to_string()
    };
    let offset_seconds = (xmax_samples as f64) / fs_f;

    Ok(CorrelationResult {
        file: file_cut,
        offset_seconds,
    })
}