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

// ============================================================================
// Tuning Constants
// ============================================================================

/// Conservative memory usage (50% of detected) - containers often over-report
const USABLE_PERCENT: usize = 50;

/// Min/Max FFT size bounds
const MIN_N_CAP: usize = 1 << 16;   // 64k
const ABS_MAX_N_CAP: usize = 1 << 26; // 64M (reduced from 268M for safety)

/// Bytes per FFT element estimation (Complex32 * 2 + overhead)
const SAFETY_BYTES_PER_ELEMENT: usize = 20;

/// Chunk size for parallel correlation (2^20 = ~1M samples = ~22s at 48kHz)
/// Keeps per-thread FFT buffers ~16MB to fit in L3 cache
const CHUNK_SIZE_SAMPLES: usize = 1 << 20;

/// Number of chunks to batch-read before parallel processing in streaming mode
/// Higher = more parallelism, but more RAM. 8 chunks = good balance.
const STREAMING_BATCH_SIZE: usize = 8;

// ============================================================================
// Memory Detection (Enhanced with Environment Variable Override)
// ============================================================================

/// Get memory limit from environment variable (in MB), if set.
/// This is the PRIMARY source of truth in containers.
fn get_env_memory_limit_bytes() -> Option<usize> {
    std::env::var("VMSAM_MEMORY_LIMIT_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|mb| mb * 1024 * 1024)
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

    // try v1
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

/// Read MemAvailable from /proc/meminfo. Fallback to MemTotal or 512MB.
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

/// Compute available memory with priority:
/// 1. VMSAM_MEMORY_LIMIT_MB env var (highest priority - explicit user setting)
/// 2. cgroup limit (container-aware)
/// 3. /proc/meminfo (fallback)
fn detect_available_memory_bytes() -> usize {
    // Priority 1: Environment variable override
    if let Some(env_limit) = get_env_memory_limit_bytes() {
        return env_limit;
    }
    
    // Priority 2: cgroup limit
    let meminfo = read_mem_available_bytes();
    if let Some(cg) = detect_cgroup_limit_bytes() {
        return std::cmp::min(cg, meminfo);
    }
    
    // Priority 3: /proc/meminfo
    meminfo
}

/// Compute usable FFT cap (power of two) based on available memory.
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

// ============================================================================
// Power-of-Two Helpers
// ============================================================================

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

fn next_pow2_floor(n: usize) -> usize {
    if n == 0 { return 1; }
    if n.is_power_of_two() { return n; }
    let mut p = 1usize;
    while p <= n { p <<= 1; }
    p >> 1
}

// ============================================================================
// FFprobe
// ============================================================================

async fn probe_samplerate_duration(path: &Path) -> Result<(u32, f64)> {
    let out = Command::new("ffprobe")
        .args(&[
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
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
        if s.is_empty() { continue; }
        if sr.is_none() && s.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(v) = s.parse::<u32>() {
                sr = Some(v);
                continue;
            }
        }
        if dur.is_none() {
            if let Ok(v) = s.parse::<f64>() {
                dur = Some(v);
            }
        }
    }
    Ok((sr.unwrap_or(48000), dur.unwrap_or(0.0)))
}

// ============================================================================
// Audio Reading (i16 format)
// ============================================================================

async fn read_full_pcm_i16(path: &Path, target_sr: u32) -> Result<(u32, Vec<i16>)> {
    let args = vec![
        "-threads".to_string(), "3".to_string(),
        "-vn".to_string(), "-nostdin".to_string(),
        "-i".to_string(), path.to_string_lossy().into_owned(),
        "-ac".to_string(), "1".to_string(),
        "-ar".to_string(), target_sr.to_string(),
        "-f".to_string(), "s16le".to_string(),
        "-acodec".to_string(), "pcm_s16le".to_string(),
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
        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok((target_sr, samples))
}

#[inline(always)]
fn i16_to_f32(s: i16) -> f32 {
    s as f32 / 32768.0
}

// ============================================================================
// FFT-based Correlation (Fast Path - Parallel Overlap-Add)
// ============================================================================

/// Correlate using parallel Overlap-Add chunked processing.
/// ref_samples: SMALLER file (reference/kernel)
/// target_samples: LARGER file (gets chunked)
fn correlate_parallel(ref_samples: &[i16], target_samples: &[i16]) -> Result<(usize, usize)> {
    let m = ref_samples.len();
    let n_target = target_samples.len();
    if m == 0 || n_target == 0 {
        anyhow::bail!("empty input for correlation");
    }

    // Chunk size: at least reference length
    let chunk_size = std::cmp::max(CHUNK_SIZE_SAMPLES, m);
    
    // FFT size based on CHUNK, not total file
    let n = next_pow2(chunk_size + m - 1);
    let total_corr_len = m + n_target - 1;

    // Prepare FFT plans (shared across threads)
    let mut planner = FftPlanner::<f32>::new();
    let fft = Arc::new(planner.plan_fft_forward(n));
    let ifft = Arc::new(planner.plan_fft_inverse(n));

    // Pre-compute reference FFT (padded to n)
    let mut ref_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    for (i, &s) in ref_samples.iter().enumerate() {
        ref_buf[i] = Complex32::new(i16_to_f32(s), 0.0);
    }
    fft.process(&mut ref_buf);
    let ref_fft = Arc::new(ref_buf);

    // Overlap-Add: step ensures no peaks missed
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
            let end = std::cmp::min(start + chunk_size, n_target);
            let chunk = &target_samples[start..end];
            let chunk_len = chunk.len();

            // Allocate LOCAL buffer (small, chunk-based)
            let mut local_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
            for (i, &s) in chunk.iter().enumerate() {
                local_buf[i] = Complex32::new(i16_to_f32(s), 0.0);
            }

            // FFT -> Multiply conjugate -> IFFT
            fft.process(&mut local_buf);
            for i in 0..n {
                let conj_ref = Complex32::new(ref_fft[i].re, -ref_fft[i].im);
                local_buf[i] *= conj_ref;
            }
            ifft.process(&mut local_buf);

            // Find max in valid region
            let valid_len = std::cmp::min(chunk_len + m - 1, n);
            let mut local_max = f32::NEG_INFINITY;
            let mut local_max_idx = 0usize;

            for i in 0..valid_len {
                let val = local_buf[i].re / (n as f32);
                let mag = val.abs();
                if mag > local_max {
                    local_max = mag;
                    local_max_idx = i;
                }
            }

            // Global correlation index
            let global_idx = start + local_max_idx;
            (local_max, global_idx)
        })
        .collect();

    // Find global maximum
    let (_, best_idx) = results
        .into_iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0.0, 0));

    Ok((total_corr_len, best_idx))
}

// ============================================================================
// Batch-Parallel Streaming Correlation (Slow Path - Memory Constrained)
// ============================================================================
// 
// Architecture: BATCH PROCESSING (chosen over Producer-Consumer)
// 
// Why Batching over Producer-Consumer:
// 1. Simpler: No need for channels, locks, or crossbeam dependency
// 2. Predictable memory: We know exactly how much RAM each batch uses
// 3. Rayon efficiency: par_iter() on batches has less overhead than channel-based
// 4. No async boundary issues: Works cleanly with tokio async context
//
// Flow:
//   Loop:
//     1. Read STREAMING_BATCH_SIZE chunks from ffmpeg stream (sequential I/O)
//     2. Process all chunks in parallel with Rayon (parallel compute)
//     3. Keep best result, discard buffers
//     4. Repeat until stream exhausted

/// Represents a chunk of audio data with its position
struct StreamChunk {
    data: Vec<i16>,
    global_start: usize,
}

/// Batch-parallel streaming correlation.
/// Reads chunks in batches, processes each batch in parallel.
async fn correlate_streaming_batch_parallel(
    ref_samples_rev: &[i16],
    stream_path: &Path,
    target_sr: u32,
    max_n_cap: usize,
) -> Result<(usize, usize)> {
    let m = ref_samples_rev.len();
    if m == 0 { anyhow::bail!("empty reference"); }

    // FFT size based on chunk, capped by memory
    let chunk_size = std::cmp::max(CHUNK_SIZE_SAMPLES, m);
    let mut n = next_pow2(chunk_size + m - 1);
    if n > max_n_cap {
        n = next_pow2_floor(max_n_cap);
        if n < m * 2 {
            anyhow::bail!("reference too large for streaming (m={}, cap={})", m, max_n_cap);
        }
    }

    // Block size for overlap-save
    let block_b = n.saturating_sub(m).saturating_add(1);
    if block_b == 0 { anyhow::bail!("computed block size is zero"); }

    // Pre-compute reference FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = Arc::new(planner.plan_fft_forward(n));
    let ifft = Arc::new(planner.plan_fft_inverse(n));

    let mut ref_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..m.min(n) {
        ref_buf[i] = Complex32::new(i16_to_f32(ref_samples_rev[i]), 0.0);
    }
    fft.process(&mut ref_buf);
    let ref_fft = Arc::new(ref_buf);

    // Spawn ffmpeg (s16le format)
    let args = vec![
        "-threads".to_string(), "3".to_string(),
        "-vn".to_string(), "-nostdin".to_string(),
        "-i".to_string(), stream_path.to_string_lossy().into_owned(),
        "-ac".to_string(), "1".to_string(),
        "-ar".to_string(), target_sr.to_string(),
        "-f".to_string(), "s16le".to_string(),
        "-acodec".to_string(), "pcm_s16le".to_string(),
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

    // Overlap buffer (reused between batches)
    let overlap_len = m.saturating_sub(1);
    let mut overlap: Vec<i16> = vec![0i16; overlap_len];
    
    // Global tracking
    let mut global_max = f32::NEG_INFINITY;
    let mut global_max_idx: usize = 0;
    let mut global_pos: usize = 0;

    // Read buffer for a single chunk
    let bytes_per_chunk = block_b * 2; // 2 bytes per i16

    loop {
        // ========== PHASE 1: Batch Read (Sequential I/O) ==========
        let mut batch: Vec<StreamChunk> = Vec::with_capacity(STREAMING_BATCH_SIZE);
        
        for _ in 0..STREAMING_BATCH_SIZE {
            let mut chunk_bytes = vec![0u8; bytes_per_chunk];
            let nread = reader.read(&mut chunk_bytes).await?;
            if nread == 0 { break; }

            let samples_read = nread / 2;
            
            // Build full input: overlap + new samples
            let mut data: Vec<i16> = Vec::with_capacity(overlap_len + samples_read);
            data.extend_from_slice(&overlap);
            for i in 0..samples_read {
                let base = i * 2;
                data.push(i16::from_le_bytes([chunk_bytes[base], chunk_bytes[base + 1]]));
            }

            batch.push(StreamChunk {
                data,
                global_start: global_pos,
            });

            // Update overlap for next chunk
            let tail_start = if samples_read >= overlap_len {
                samples_read - overlap_len
            } else {
                0
            };
            for i in 0..overlap_len {
                let src_idx = tail_start + i;
                if src_idx < samples_read {
                    let base = src_idx * 2;
                    overlap[i] = i16::from_le_bytes([chunk_bytes[base], chunk_bytes[base + 1]]);
                }
            }

            global_pos += samples_read;
        }

        if batch.is_empty() { break; }

        // ========== PHASE 2: Batch Process (Parallel Compute) ==========
        let batch_results: Vec<(f32, usize)> = batch
            .par_iter()
            .map(|chunk| {
                let chunk_data = &chunk.data;
                
                // Allocate local FFT buffer
                let mut local_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
                for (i, &s) in chunk_data.iter().enumerate().take(n) {
                    local_buf[i] = Complex32::new(i16_to_f32(s), 0.0);
                }

                // FFT -> Multiply -> IFFT
                fft.process(&mut local_buf);
                for i in 0..n {
                    local_buf[i] = local_buf[i] * ref_fft[i];
                }
                ifft.process(&mut local_buf);

                // Find max in output region (after overlap, valid correlation)
                let start_idx = m.saturating_sub(1);
                let samples_in_chunk = chunk_data.len().saturating_sub(overlap_len);
                
                let mut local_max = f32::NEG_INFINITY;
                let mut local_max_offset = 0usize;

                for i in 0..samples_in_chunk {
                    let idx = start_idx + i;
                    if idx >= n { break; }
                    let val = local_buf[idx].re / (n as f32);
                    let mag = val.abs();
                    if mag > local_max {
                        local_max = mag;
                        local_max_offset = i;
                    }
                }

                (local_max, chunk.global_start + local_max_offset)
            })
            .collect();

        // Update global max from this batch
        for (mag, idx) in batch_results {
            if mag > global_max {
                global_max = mag;
                global_max_idx = idx;
            }
        }
    }

    let _ = child.wait().await;
    Ok((n, global_max_idx))
}

// ============================================================================
// Public API
// ============================================================================

/// Top-level correlation function: parallel if memory allows, batch-streaming otherwise.
pub async fn second_correlation_async(in1: &str, in2: &str, _pool_capacity: usize) -> Result<CorrelationResult> {
    // Probe both files
    let p1 = Path::new(in1);
    let p2 = Path::new(in2);
    let (sr1, dur1) = probe_samplerate_duration(p1).await?;
    let (sr2, dur2) = probe_samplerate_duration(p2).await?;

    let target_sr = std::cmp::min(sr1, sr2);
    let est1 = (sr1 as f64 * dur1).round() as usize;
    let est2 = (sr2 as f64 * dur2).round() as usize;

    // Memory calculation
    let storage_bytes = (est1 + est2) * 2;
    let num_threads = rayon::current_num_threads();
    let ref_est = std::cmp::min(est1, est2);
    let chunk_fft_size = next_pow2(CHUNK_SIZE_SAMPLES + ref_est);
    let per_thread_fft_bytes = chunk_fft_size * 8 * 2;
    let total_fft_bytes = num_threads * per_thread_fft_bytes;
    let bytes_needed = storage_bytes + total_fft_bytes;

    let avail = detect_available_memory_bytes();
    let usable = avail.saturating_mul(USABLE_PERCENT) / 100;

    if bytes_needed <= usable {
        // ========== FAST PATH: Load both files, parallel correlation ==========
        let (_sr_a, a) = read_full_pcm_i16(p1, target_sr).await?;
        let (_sr_b, b) = read_full_pcm_i16(p2, target_sr).await?;

        // CRITICAL: Smaller vector as reference for efficient chunking
        let (ref_samples, target_samples, ref_is_file1) = if a.len() <= b.len() {
            (&a, &b, true)
        } else {
            (&b, &a, false)
        };

        let (total_len, xmax) = correlate_parallel(ref_samples, target_samples)?;
        let fs_f = target_sr as f64;
        
        let (file_cut, offset_seconds) = if xmax > total_len / 2 {
            let offset = (total_len - xmax) as f64 / fs_f;
            if ref_is_file1 { (in2.to_string(), offset) } else { (in1.to_string(), offset) }
        } else {
            let offset = xmax as f64 / fs_f;
            if ref_is_file1 { (in1.to_string(), offset) } else { (in2.to_string(), offset) }
        };
        return Ok(CorrelationResult { file: file_cut, offset_seconds });
    }

    // ========== SLOW PATH: Batch-Parallel Streaming ==========
    let max_n_cap = compute_max_n_cap();
    
    // Smaller file as reference
    let (ref_path, stream_path, ref_is_file1) = if est1 <= est2 {
        (p1, p2, true)
    } else {
        (p2, p1, false)
    };

    // Read reference fully
    let (_sr_ref, mut ref_samples) = read_full_pcm_i16(ref_path, target_sr).await?;
    if ref_samples.is_empty() {
        anyhow::bail!("reference empty after read");
    }

    // Reverse reference for convolution
    ref_samples.reverse();
    let m = ref_samples.len();
    if m > max_n_cap {
        anyhow::bail!("reference too large (m={}, cap={})", m, max_n_cap);
    }

    // Run batch-parallel streaming
    let (_padsize, xmax_samples) = correlate_streaming_batch_parallel(
        &ref_samples, stream_path, target_sr, max_n_cap
    ).await?;

    let fs_f = target_sr as f64;
    let file_cut = if ref_is_file1 { in2.to_string() } else { in1.to_string() };
    let offset_seconds = (xmax_samples as f64) / fs_f;

    Ok(CorrelationResult { file: file_cut, offset_seconds })
}

// ============================================================================
// Tests
// ============================================================================

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
        assert_eq!(next_pow2(1025), 2048);
    }

    #[test]
    fn test_env_memory_override() {
        // This test just verifies the function doesn't panic
        let _ = get_env_memory_limit_bytes();
    }

    #[test]
    fn test_correlate_parallel_simple() {
        let reference: Vec<i16> = vec![1000, 2000, 3000, 2000, 1000];
        let mut target: Vec<i16> = vec![0i16; 100];
        for (i, &v) in reference.iter().enumerate() {
            target[50 + i] = v;
        }

        let result = correlate_parallel(&reference, &target);
        assert!(result.is_ok());
        let (_n, best_idx) = result.unwrap();
        let expected = 50 + reference.len() - 1;
        assert!((best_idx as isize - expected as isize).abs() <= 2);
    }

    #[test]
    fn test_memory_detection_runs() {
        let mem = detect_available_memory_bytes();
        assert!(mem > 0);
    }
}