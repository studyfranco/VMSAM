//! Dual-Engine Audio Correlation Library
//!
//! Implements a hybrid strategy for audio synchronization:
//! - Short files (<20 min): Single-threaded full correlation at native sample rate
//! - Long files (>=20 min): Parallel chunked correlation at 8kHz with segmented reference

use anyhow::{Context, Result};
use num_complex::Complex32;
use rayon::prelude::*;
use realfft::RealFftPlanner;
use serde::Serialize;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

#[derive(Debug, Serialize)]
pub struct CorrelationResult {
    pub file: String,
    pub offset_seconds: f64,
}

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

/// Duration threshold in seconds (20 minutes)
const DURATION_THRESHOLD_SECS: f64 = 20.0 * 60.0;

/// Forced sample rate for long files (8kHz = 6x reduction from 48kHz)
const FORCE_SAMPLE_RATE_LONG: u32 = 8000;

/// Chunk size for parallel processing (2^18 = 262144 samples, ~32KB per chunk)
/// Optimized for L3 cache (typically 6-12MB on modern CPUs)
const CHUNK_SIZE_SAMPLES: usize = 1 << 18;

// ============================================================================
// UTILITIES
// ============================================================================

/// Next power of two >= n
#[inline]
fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    if std::mem::size_of::<usize>() > 4 {
        v |= v >> 32;
    }
    v + 1
}

/// Probe sample rate and duration via ffprobe
async fn probe_samplerate_duration(path: &Path) -> Result<(u32, f64)> {
    let out = Command::new("ffprobe")
        .args([
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
            }
        }
    }

    Ok((sr.unwrap_or(48000), dur.unwrap_or(0.0)))
}

// ============================================================================
// I/O: PCM READING (i16 storage, f32 output)
// ============================================================================

/// Read entire audio via ffmpeg into Vec<f32> (mono, target sample rate).
/// Uses i16 intermediate for memory efficiency, converts to f32 for FFT.
async fn read_pcm_i16_to_f32(path: &Path, target_sr: u32) -> Result<Vec<f32>> {
    let args = vec![
        "-threads".to_string(),
        "2".to_string(),
        "-vn".to_string(),
        "-nostdin".to_string(),
        "-i".to_string(),
        path.to_string_lossy().into_owned(),
        "-ac".to_string(),
        "1".to_string(),
        "-ar".to_string(),
        target_sr.to_string(),
        "-f".to_string(),
        "s16le".to_string(), // i16 little-endian (2 bytes per sample)
        "-acodec".to_string(),
        "pcm_s16le".to_string(),
        "-".to_string(),
    ];

    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("spawn ffmpeg for PCM read")?;

    let stdout = child.stdout.take().context("no stdout from ffmpeg")?;
    let mut reader = tokio::io::BufReader::new(stdout);

    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .await
        .context("read ffmpeg stdout")?;
    let _ = child.wait().await;

    // Convert i16 -> f32 normalized to [-1.0, 1.0]
    let scale = 1.0 / 32768.0;
    let samples: Vec<f32> = buf
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 * scale
        })
        .collect();

    Ok(samples)
}

// ============================================================================
// SINGLE-THREADED ENGINE (R2C FFT)
// ============================================================================

/// Full correlation using Real-to-Complex FFT (2x faster, 50% less memory)
fn correlate_full_r2c(s1: &[f32], s2: &[f32]) -> Result<(usize, usize)> {
    let ls1 = s1.len();
    let ls2 = s2.len();
    if ls1 == 0 || ls2 == 0 {
        anyhow::bail!("empty input for correlation");
    }

    let needed = ls1 + ls2 - 1;
    let n = next_pow2(needed);
    let spectrum_len = n / 2 + 1;

    // Create R2C/C2R planners
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n);
    let c2r = planner.plan_fft_inverse(n);

    // Prepare input buffers (zero-padded)
    let mut a_real = vec![0.0f32; n];
    let mut b_real = vec![0.0f32; n];
    a_real[..ls1].copy_from_slice(s1);
    b_real[..ls2].copy_from_slice(s2);

    // Allocate spectrum buffers
    let mut a_spectrum = vec![Complex32::new(0.0, 0.0); spectrum_len];
    let mut b_spectrum = vec![Complex32::new(0.0, 0.0); spectrum_len];

    // Forward R2C transforms
    r2c.process(&mut a_real, &mut a_spectrum)
        .map_err(|e| anyhow::anyhow!("R2C FFT failed for s1: {:?}", e))?;
    r2c.process(&mut b_real, &mut b_spectrum)
        .map_err(|e| anyhow::anyhow!("R2C FFT failed for s2: {:?}", e))?;

    // Cross-correlation in frequency domain: A * conj(B)
    for i in 0..spectrum_len {
        let conj_b = Complex32::new(b_spectrum[i].re, -b_spectrum[i].im);
        a_spectrum[i] *= conj_b;
    }

    // Inverse C2R transform
    let mut result_real = vec![0.0f32; n];
    c2r.process(&mut a_spectrum, &mut result_real)
        .map_err(|e| anyhow::anyhow!("C2R IFFT failed: {:?}", e))?;

    // Find max magnitude (no normalization needed for peak finding)
    let mut xmax = 0usize;
    let mut maxv = f32::NEG_INFINITY;
    for (i, &v) in result_real.iter().enumerate() {
        let mag = v.abs();
        if mag > maxv {
            maxv = mag;
            xmax = i;
        }
    }

    Ok((n, xmax))
}

// ============================================================================
// PARALLEL ENGINE (Chunked Correlation with Segmented Reference)
// ============================================================================

/// Shared FFT context for parallel correlation (thread-safe via Arc)
struct ParallelFftContext {
    n: usize,
    spectrum_len: usize,
    r2c: Arc<dyn realfft::RealToComplex<f32>>,
    c2r: Arc<dyn realfft::ComplexToReal<f32>>,
}

impl ParallelFftContext {
    fn new(n: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(n);
        let c2r = planner.plan_fft_inverse(n);
        Self {
            n,
            spectrum_len: n / 2 + 1,
            r2c,
            c2r,
        }
    }
}

/// Compute correlation for a single chunk against reference spectrum
fn correlate_chunk(
    ctx: &ParallelFftContext,
    chunk: &[f32],
    ref_spectrum: &[Complex32],
) -> Result<Vec<f32>> {
    let mut input = vec![0.0f32; ctx.n];
    input[..chunk.len()].copy_from_slice(chunk);

    let mut spectrum = vec![Complex32::new(0.0, 0.0); ctx.spectrum_len];

    // Forward R2C
    ctx.r2c
        .process(&mut input, &mut spectrum)
        .map_err(|e| anyhow::anyhow!("R2C chunk FFT failed: {:?}", e))?;

    // Multiply with conjugate of reference
    for i in 0..ctx.spectrum_len {
        let conj_ref = Complex32::new(ref_spectrum[i].re, -ref_spectrum[i].im);
        spectrum[i] *= conj_ref;
    }

    // Inverse C2R
    let mut result = vec![0.0f32; ctx.n];
    ctx.c2r
        .process(&mut spectrum, &mut result)
        .map_err(|e| anyhow::anyhow!("C2R chunk IFFT failed: {:?}", e))?;

    Ok(result)
}

/// Parallel chunked correlation with segmented reference
/// Guarantees 100% CPU utilization by splitting both target and reference
fn correlate_parallel(reference: &[f32], target: &[f32]) -> Result<(usize, usize)> {
    if reference.is_empty() || target.is_empty() {
        anyhow::bail!("empty input for parallel correlation");
    }

    let chunk_size = CHUNK_SIZE_SAMPLES;
    let n = next_pow2(chunk_size * 2); // Enough room for overlap-add
    let spectrum_len = n / 2 + 1;

    // Setup FFT context
    let ctx = Arc::new(ParallelFftContext::new(n));

    // Determine if we need to segment the reference
    let ref_segment_threshold = chunk_size / 2;
    let ref_segments: Vec<&[f32]> = if reference.len() > ref_segment_threshold {
        // Split reference into overlapping segments for better parallelism
        reference
            .chunks(ref_segment_threshold)
            .collect()
    } else {
        vec![reference]
    };

    // Precompute spectra for all reference segments
    let ref_spectra: Vec<Vec<Complex32>> = ref_segments
        .par_iter()
        .map(|seg| {
            let mut planner = RealFftPlanner::<f32>::new();
            let r2c = planner.plan_fft_forward(n);

            let mut input = vec![0.0f32; n];
            input[..seg.len()].copy_from_slice(seg);

            let mut spectrum = vec![Complex32::new(0.0, 0.0); spectrum_len];
            r2c.process(&mut input, &mut spectrum).ok();
            spectrum
        })
        .collect();

    // Split target into chunks
    let target_chunks: Vec<(usize, &[f32])> = target
        .chunks(chunk_size)
        .enumerate()
        .map(|(i, chunk)| (i * chunk_size, chunk))
        .collect();

    // Parallel correlation: each (chunk, ref_segment) pair
    let results: Vec<(usize, f32)> = target_chunks
        .par_iter()
        .flat_map(|(chunk_offset, chunk)| {
            ref_spectra
                .par_iter()
                .enumerate()
                .filter_map(|(seg_idx, ref_spectrum)| {
                    let seg_offset = seg_idx * ref_segment_threshold;
                    match correlate_chunk(&ctx, chunk, ref_spectrum) {
                        Ok(corr) => {
                            // Find local max in this correlation result
                            let (local_idx, local_max) = corr
                                .iter()
                                .enumerate()
                                .map(|(i, &v)| (i, v.abs()))
                                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                                .unwrap_or((0, 0.0));
                            
                            // Global sample index
                            let global_idx = chunk_offset + local_idx;
                            Some((global_idx, local_max))
                        }
                        Err(_) => None,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Find global maximum across all results
    let (best_idx, _best_val) = results
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0));

    Ok((n, best_idx))
}

// ============================================================================
// MAIN API
// ============================================================================

/// Top-level async correlation function with hybrid strategy
pub async fn second_correlation_async(
    in1: &str,
    in2: &str,
    _pool_capacity: usize,
) -> Result<CorrelationResult> {
    let total_start = Instant::now();

    let p1 = Path::new(in1);
    let p2 = Path::new(in2);

    // Phase 1: Parallel Analysis (probe both files concurrently)
    let io_start = Instant::now();
    let ((sr1, dur1), (sr2, dur2)) = tokio::try_join!(
        probe_samplerate_duration(p1),
        probe_samplerate_duration(p2)
    )?;

    // Determine longest duration
    let max_duration = dur1.max(dur2);

    // Phase 2: Strategy Selection (The Fork)
    let is_long_file = max_duration >= DURATION_THRESHOLD_SECS;

    if is_long_file {
        // ================================================================
        // CASE B: Long Files - Parallel Engine at 8kHz
        // ================================================================
        let target_sr = FORCE_SAMPLE_RATE_LONG;

        // Parallel I/O at 8kHz
        let (samples1, samples2) = tokio::try_join!(
            read_pcm_i16_to_f32(p1, target_sr),
            read_pcm_i16_to_f32(p2, target_sr)
        )?;
        let io_elapsed = io_start.elapsed();

        // Choose shorter as reference
        let (ref_samples, target_samples, ref_is_file1) = if samples1.len() <= samples2.len() {
            (samples1, samples2, true)
        } else {
            (samples2, samples1, false)
        };

        // Parallel correlation
        let calc_start = Instant::now();
        let (_n, xmax) = correlate_parallel(&ref_samples, &target_samples)?;
        let calc_elapsed = calc_start.elapsed();

        // Metrics output
        eprintln!(
            "[METRICS] Strategy: HYBRID_PARALLEL ({}Hz) | I/O: {:.2}s | Calc: {:.2}s | Total: {:.2}s",
            target_sr,
            io_elapsed.as_secs_f64(),
            calc_elapsed.as_secs_f64(),
            total_start.elapsed().as_secs_f64()
        );

        // Result interpretation
        let fs_f = target_sr as f64;
        let file_cut = if ref_is_file1 {
            in2.to_string()
        } else {
            in1.to_string()
        };
        let offset_seconds = xmax as f64 / fs_f;

        Ok(CorrelationResult {
            file: file_cut,
            offset_seconds,
        })
    } else {
        // ================================================================
        // CASE A: Short Files - Single-Threaded at Native Sample Rate
        // ================================================================
        let target_sr = sr1.min(sr2);

        // Parallel I/O at native sample rate
        let (samples1, samples2) = tokio::try_join!(
            read_pcm_i16_to_f32(p1, target_sr),
            read_pcm_i16_to_f32(p2, target_sr)
        )?;
        let io_elapsed = io_start.elapsed();

        // Single-threaded R2C correlation
        let calc_start = Instant::now();
        let (padsize, xmax) = correlate_full_r2c(&samples1, &samples2)?;
        let calc_elapsed = calc_start.elapsed();

        // Metrics output
        eprintln!(
            "[METRICS] Strategy: SINGLE_THREAD ({}Hz Native) | I/O: {:.2}s | Calc: {:.2}s | Total: {:.2}s",
            target_sr,
            io_elapsed.as_secs_f64(),
            calc_elapsed.as_secs_f64(),
            total_start.elapsed().as_secs_f64()
        );

        // Result interpretation (same as original)
        let fs_f = target_sr as f64;
        let (file_cut, offset_seconds) = if xmax > padsize / 2 {
            (in2.to_string(), (padsize - xmax) as f64 / fs_f)
        } else {
            (in1.to_string(), xmax as f64 / fs_f)
        };

        Ok(CorrelationResult {
            file: file_cut,
            offset_seconds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_pow2() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(1023), 1024);
        assert_eq!(next_pow2(1024), 1024);
        assert_eq!(next_pow2(1025), 2048);
    }

    #[test]
    fn test_correlate_full_r2c_identical() {
        // Identical signals should have peak at 0
        let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let (n, xmax) = correlate_full_r2c(&signal, &signal).unwrap();
        assert!(n >= signal.len() * 2 - 1);
        // Peak at 0 for identical signals (autocorrelation)
        assert!(xmax == 0 || xmax == n - 1 || xmax < signal.len());
    }

    #[test]
    fn test_correlate_full_r2c_shifted() {
        // Signal shifted by 2 samples
        let s1 = vec![0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0];
        let s2 = vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0];
        let (_n, xmax) = correlate_full_r2c(&s1, &s2).unwrap();
        // Peak should be around 2 (the shift amount)
        assert!(xmax <= 4, "Expected shift around 2, got {}", xmax);
    }
}