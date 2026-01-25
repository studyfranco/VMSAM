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
const USABLE_PERCENT: usize = 85; // use 85% of detected available memory
const MIN_N_CAP: usize = 1 << 16; // min FFT size (64k)
const ABS_MAX_N_CAP: usize = 1 << 28; // hard cap for FFT size (~268M)
const SAFETY_BYTES_PER_ELEMENT: usize = 18; // bytes per FFT "element" estimation

/// Chunk size for parallel correlation (2^20 = ~1M samples = ~22 seconds at 48kHz)
/// This keeps per-thread FFT buffers small (~16MB each) to fit in L3 cache
const CHUNK_SIZE_SAMPLES: usize = 1 << 20; // 1,048,576 samples

// ----------------------------------------------------------------------------
// Memory Detection (PRESERVED from original)
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Power-of-Two Helpers
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// FFprobe
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Audio Reading (i16 format for memory efficiency)
// ----------------------------------------------------------------------------

/// Read entire audio via ffmpeg into Vec<i16> (mono, target sample rate).
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
    reader.read_to_end(&mut buf).await.context("read ffmpeg stdout")?;
    let _ = child.wait().await;

    let mut samples = Vec::with_capacity(buf.len() / 2);
    for chunk in buf.chunks_exact(2) {
        let bytes = [chunk[0], chunk[1]];
        samples.push(i16::from_le_bytes(bytes));
    }
    Ok((target_sr, samples))
}

/// Convert i16 sample to f32 (normalized to [-1, 1])
#[inline(always)]
fn i16_to_f32(s: i16) -> f32 {
    s as f32 / 32768.0
}

// ----------------------------------------------------------------------------
// FFT-based Correlation (Parallelized with Rayon + Overlap-Add)
// ----------------------------------------------------------------------------

/// Correlate using parallel Overlap-Add chunked processing.
/// 
/// CRITICAL: FFT size `n` is based on CHUNK_SIZE + ref_len, NOT total file size.
/// This ensures each thread allocates small buffers (~32-64MB) regardless of file size.
/// 
/// ref_samples: reference audio (i16)
/// target_samples: target audio (i16)
/// Returns (total_correlation_length, best_lag_samples)
fn correlate_parallel(ref_samples: &[i16], target_samples: &[i16]) -> Result<(usize, usize)> {
    let m = ref_samples.len();
    let n_target = target_samples.len();
    if m == 0 || n_target == 0 {
        anyhow::bail!("empty input for correlation");
    }

    // Use fixed chunk size, but ensure it's at least as large as reference
    let chunk_size = std::cmp::max(CHUNK_SIZE_SAMPLES, m);
    
    // FFT size based on CHUNK, not total file!
    // n = next_pow2(chunk_size + m - 1) for linear convolution via FFT
    let n = next_pow2(chunk_size + m - 1);
    
    // Total correlation length (for interpreting final result)
    let total_corr_len = m + n_target - 1;

    // Prepare FFT planner
    let mut planner = FftPlanner::<f32>::new();
    let fft = Arc::new(planner.plan_fft_forward(n));
    let ifft = Arc::new(planner.plan_fft_inverse(n));

    // Pre-compute reference FFT (padded to chunk-based n)
    let mut ref_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    for (i, &s) in ref_samples.iter().enumerate() {
        ref_buf[i] = Complex32::new(i16_to_f32(s), 0.0);
    }
    fft.process(&mut ref_buf);
    let ref_fft = Arc::new(ref_buf);

    // Overlap-Add: step size ensures no peaks are missed at boundaries
    // overlap = m - 1 samples, so step = chunk_size - (m - 1)
    let overlap = m.saturating_sub(1);
    let step = chunk_size.saturating_sub(overlap);
    if step == 0 {
        anyhow::bail!("chunk_size too small for reference length");
    }

    // Generate chunk start positions
    let mut chunk_starts: Vec<usize> = Vec::new();
    let mut pos = 0;
    while pos < n_target {
        chunk_starts.push(pos);
        pos += step;
    }

    // Process chunks in parallel
    let results: Vec<(f32, usize)> = chunk_starts
        .par_iter()
        .map(|&start| {
            // Determine chunk bounds
            let end = std::cmp::min(start + chunk_size, n_target);
            let chunk = &target_samples[start..end];
            let chunk_len = chunk.len();

            // Allocate LOCAL buffer of size n (small, chunk-based!)
            let mut local_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
            
            // Fill with chunk samples
            for (i, &s) in chunk.iter().enumerate() {
                local_buf[i] = Complex32::new(i16_to_f32(s), 0.0);
            }

            // FFT of chunk
            fft.process(&mut local_buf);

            // Multiply with conjugate of reference FFT (correlation in freq domain)
            for i in 0..n {
                let conj_ref = Complex32::new(ref_fft[i].re, -ref_fft[i].im);
                local_buf[i] *= conj_ref;
            }

            // Inverse FFT
            ifft.process(&mut local_buf);

            // Find max in valid correlation output region
            // Valid indices: 0 to (chunk_len + m - 2), but capped at n
            let valid_len = std::cmp::min(chunk_len + m - 1, n);
            
            let mut local_max = f32::NEG_INFINITY;
            let mut local_max_idx = 0usize;

            for i in 0..valid_len {
                // Normalize by n (IFFT scaling)
                let val = local_buf[i].re / (n as f32);
                let mag = val.abs();
                if mag > local_max {
                    local_max = mag;
                    local_max_idx = i;
                }
            }

            // Translate local index to GLOBAL sample position
            // The correlation output at local index `i` corresponds to:
            // - lag = i - (m - 1) relative to chunk start
            // - global_lag = start + i - (m - 1) relative to target start
            // But we want the raw correlation index (as if we did full correlation)
            // global_correlation_index = start + local_max_idx
            let global_idx = start + local_max_idx;

            (local_max, global_idx)
        })
        .collect();

    // Find global maximum across all chunks
    let (_, best_idx) = results
        .into_iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0.0, 0));

    Ok((total_corr_len, best_idx))
}

/// Overlap-save correlation streaming for memory-constrained scenarios.
async fn correlate_overlap_save_i16(
    ref_samples_rev: &[i16],
    stream_path: &Path,
    target_sr: u32,
    max_n_cap: usize,
) -> Result<(usize, usize)> {
    let m = ref_samples_rev.len();
    if m == 0 { anyhow::bail!("empty reference"); }

    // Choose n: based on reasonable chunk, not total file
    let chunk_size = std::cmp::max(CHUNK_SIZE_SAMPLES, m);
    let mut n = next_pow2(chunk_size + m - 1);
    if n > max_n_cap {
        n = next_pow2_floor(max_n_cap);
        if n < m * 2 {
            anyhow::bail!("reference too large for streaming FFT (m={}, cap={})", m, max_n_cap);
        }
    }

    // Block size B = N - m + 1
    let block_b = n.saturating_sub(m).saturating_add(1);
    if block_b == 0 { anyhow::bail!("computed block size is zero"); }

    // Prepare planner and precompute ref_fft
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut ref_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..m.min(n) {
        ref_buf[i] = Complex32::new(i16_to_f32(ref_samples_rev[i]), 0.0);
    }
    fft.process(&mut ref_buf);
    let ref_fft = ref_buf;

    // Spawn ffmpeg (s16le format)
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
        .context("spawn ffmpeg for streaming")?;

    let stdout = child.stdout.take().context("no stdout from ffmpeg")?;
    let mut reader = tokio::io::BufReader::new(stdout);

    // Buffers for overlap-save
    let overlap_len = m.saturating_sub(1);
    let mut overlap: Vec<i16> = vec![0i16; overlap_len];
    let mut local_bytes: Vec<u8> = vec![0u8; block_b.saturating_mul(2)];
    let mut in_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];

    let mut maxv = f32::NEG_INFINITY;
    let mut max_idx_samples: usize = 0;
    let mut global_pos: usize = 0;

    loop {
        let nread = reader.read(&mut local_bytes).await?;
        if nread == 0 { break; }
        let samples_read = nread / 2;

        // Zero buffer
        for v in in_buf.iter_mut() { *v = Complex32::new(0.0, 0.0); }

        // Copy overlap
        for (i, &v) in overlap.iter().enumerate() {
            in_buf[i] = Complex32::new(i16_to_f32(v), 0.0);
        }

        // Copy new samples
        for i in 0..samples_read {
            let base = i * 2;
            let bytes = [local_bytes[base], local_bytes[base+1]];
            let samp = i16::from_le_bytes(bytes);
            let idx = overlap.len() + i;
            if idx < in_buf.len() {
                in_buf[idx] = Complex32::new(i16_to_f32(samp), 0.0);
            }
        }

        // FFT
        fft.process(&mut in_buf);

        // Multiply with ref_fft
        for i in 0..n {
            in_buf[i] = in_buf[i] * ref_fft[i];
        }

        // Inverse FFT
        ifft.process(&mut in_buf);

        // Find max in output region
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

        // Update overlap
        let mut tail: Vec<i16> = Vec::with_capacity(overlap_len + samples_read);
        tail.extend_from_slice(&overlap);
        for i in 0..samples_read {
            let base = i * 2;
            let bytes = [local_bytes[base], local_bytes[base+1]];
            tail.push(i16::from_le_bytes(bytes));
        }
        if tail.len() >= overlap_len {
            let start = tail.len().saturating_sub(overlap_len);
            overlap.clear();
            overlap.extend_from_slice(&tail[start..]);
        } else {
            let mut new_ov = vec![0i16; overlap_len];
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

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

/// Top-level correlation function: parallel if memory allows, streaming otherwise.
pub async fn second_correlation_async(in1: &str, in2: &str, _pool_capacity: usize) -> Result<CorrelationResult> {
    // Probe both files
    let p1 = Path::new(in1);
    let p2 = Path::new(in2);
    let (sr1, dur1) = probe_samplerate_duration(p1).await?;
    let (sr2, dur2) = probe_samplerate_duration(p2).await?;

    // Choose common target sr
    let target_sr = std::cmp::min(sr1, sr2);

    // Estimate samples
    let est1 = (sr1 as f64 * dur1).round() as usize;
    let est2 = (sr2 as f64 * dur2).round() as usize;

    // Memory check: can we load both files as i16?
    // Each sample = 2 bytes, plus FFT overhead per-thread
    let storage_bytes = (est1 + est2) * 2;
    let num_threads = rayon::current_num_threads();
    let chunk_fft_size = next_pow2(CHUNK_SIZE_SAMPLES + std::cmp::min(est1, est2));
    let per_thread_fft_bytes = chunk_fft_size * 8 * 2; // Complex32 = 8 bytes, 2 buffers
    let total_fft_bytes = num_threads * per_thread_fft_bytes;
    let bytes_needed = storage_bytes + total_fft_bytes;

    let avail = detect_available_memory_bytes();
    let usable = avail.saturating_mul(USABLE_PERCENT) / 100;

    if bytes_needed <= usable {
        // Fast path: load both files fully (as i16) and run parallel correlation
        let (_sr_a, a) = read_full_pcm_i16(p1, target_sr).await?;
        let (_sr_b, b) = read_full_pcm_i16(p2, target_sr).await?;

        // CRITICAL: Always use SMALLER vector as reference for efficient chunking
        // The larger vector gets split into parallel chunks, the smaller one is the FFT kernel
        let (ref_samples, target_samples, ref_is_file1) = if a.len() <= b.len() {
            (&a, &b, true)  // a is reference (file1)
        } else {
            (&b, &a, false) // b is reference (file2)
        };

        let (total_len, xmax) = correlate_parallel(ref_samples, target_samples)?;
        let fs_f = target_sr as f64;
        
        // Determine which file to cut based on correlation peak position
        // Account for whether we swapped the inputs
        let (file_cut, offset_seconds) = if xmax > total_len / 2 {
            let offset = (total_len - xmax) as f64 / fs_f;
            if ref_is_file1 {
                (in2.to_string(), offset)
            } else {
                (in1.to_string(), offset)
            }
        } else {
            let offset = xmax as f64 / fs_f;
            if ref_is_file1 {
                (in1.to_string(), offset)
            } else {
                (in2.to_string(), offset)
            }
        };
        return Ok(CorrelationResult { file: file_cut, offset_seconds });
    }

    // Slow path: streaming for memory-constrained scenarios
    let max_n_cap = compute_max_n_cap();
    
    // Choose shorter file as reference
    let (ref_path, stream_path, _ref_est) = if est1 <= est2 {
        (p1, p2, est1)
    } else {
        (p2, p1, est2)
    };

    // Read reference fully (as i16)
    let (_sr_ref, mut ref_samples) = read_full_pcm_i16(ref_path, target_sr).await?;
    if ref_samples.is_empty() {
        anyhow::bail!("reference empty after read");
    }

    // Reverse reference for convolution
    ref_samples.reverse();
    let m = ref_samples.len();
    if m > max_n_cap {
        anyhow::bail!(
            "reference too large for streaming (m={}, cap={})",
            m, max_n_cap
        );
    }

    // Run overlap-save streaming
    let (_padsize, xmax_samples) = correlate_overlap_save_i16(&ref_samples, stream_path, target_sr, max_n_cap).await?;

    let fs_f = target_sr as f64;
    let file_cut = if ref_path == p1 { in2.to_string() } else { in1.to_string() };
    let offset_seconds = (xmax_samples as f64) / fs_f;

    Ok(CorrelationResult { file: file_cut, offset_seconds })
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i16_to_f32_conversion() {
        assert!((i16_to_f32(0) - 0.0).abs() < 1e-6);
        assert!((i16_to_f32(32767) - 0.999969).abs() < 1e-4);
        assert!((i16_to_f32(-32768) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_next_pow2() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(1024), 1024);
        assert_eq!(next_pow2(1025), 2048);
    }

    #[test]
    fn test_next_pow2_floor() {
        assert_eq!(next_pow2_floor(1), 1);
        assert_eq!(next_pow2_floor(2), 2);
        assert_eq!(next_pow2_floor(3), 2);
        assert_eq!(next_pow2_floor(5), 4);
        assert_eq!(next_pow2_floor(1024), 1024);
        assert_eq!(next_pow2_floor(1025), 1024);
    }

    #[test]
    fn test_chunk_fft_size_is_bounded() {
        // Verify FFT size is based on CHUNK, not arbitrary large files
        let ref_len = 1_000_000; // 1M samples reference
        let chunk_size = std::cmp::max(CHUNK_SIZE_SAMPLES, ref_len);
        let n = next_pow2(chunk_size + ref_len - 1);
        
        // Should be ~8M (2^23), NOT 2^28 or larger
        assert!(n <= 1 << 24, "FFT size {} is too large!", n);
        assert!(n >= chunk_size, "FFT size {} is too small!", n);
    }

    #[test]
    fn test_correlate_parallel_simple() {
        // Simple test: find where a pattern appears in a signal
        let reference: Vec<i16> = vec![1000, 2000, 3000, 2000, 1000];
        let mut target: Vec<i16> = vec![0i16; 100];
        // Place reference at offset 50
        for (i, &v) in reference.iter().enumerate() {
            target[50 + i] = v;
        }

        let result = correlate_parallel(&reference, &target);
        assert!(result.is_ok());
        let (_n, best_idx) = result.unwrap();
        // The correlation peak should be around offset 50 + (ref_len - 1) = 54
        // (correlation index = target_offset + ref_len - 1 for perfect match)
        let expected = 50 + reference.len() - 1;
        assert!((best_idx as isize - expected as isize).abs() <= 2, 
            "Expected peak near {}, got {}", expected, best_idx);
    }

    #[test]
    fn test_memory_detection_runs() {
        let mem = detect_available_memory_bytes();
        assert!(mem > 0);
    }

    #[test]
    fn test_compute_max_n_cap_runs() {
        let cap = compute_max_n_cap();
        assert!(cap >= MIN_N_CAP);
        assert!(cap <= ABS_MAX_N_CAP);
        assert!(cap.is_power_of_two());
    }
}