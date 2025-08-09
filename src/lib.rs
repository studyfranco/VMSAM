use anyhow::{Context, Result};
use num_complex::Complex32;
use rustfft::FftPlanner;
use serde::Serialize;
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
const BLOCK_SIZE_SAMPLES: usize = 131_072; // ~0.5 MB of f32 (131k * 4 = 524kB)
const MAX_N_CAP: usize = 1 << 22; // cap for FFT size (~4M). Increase if you have more RAM.

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

/// Spawn ffmpeg child that outputs mono pcm_f32le to stdout.
/// Caller must keep the returned Child alive while reading child.stdout().
fn spawn_ffmpeg_child(path: &Path, pan_expr: Option<String>, target_sr: Option<u32>) -> Result<Child> {
    // build args as Vec<String> so we don't borrow temporaries
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
/// Returns (used_sample_rate, samples).
async fn read_all_pcm_f32_from_ffmpeg(path: &Path, pan_expr: Option<String>, target_sr: Option<u32>) -> Result<(u32, Vec<f32>)> {
    let (sr0, _dur) = probe_samplerate_duration(path).await?;
    let target_sr = target_sr.unwrap_or(sr0);

    // build args similar to spawn_ffmpeg_child
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

    // wait the child to finish (collect status)
    let _ = child.wait().await;

    // convert bytes -> f32 little endian
    let mut samples = Vec::with_capacity(buf.len() / 4);
    for chunk in buf.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        samples.push(f32::from_le_bytes(bytes));
    }
    Ok((target_sr, samples))
}

/// next power of two
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

/// Core: streaming cross-correlation using overlap-save.
/// - load the smaller file fully as reference (rev)
/// - stream the larger file in blocks, compute convolution with reversed ref via FFT
pub async fn second_correlation_streaming(in1: &str, in2: &str, pool_capacity: usize) -> Result<CorrelationResult> {
    // probe both files
    let p1 = Path::new(in1);
    let p2 = Path::new(in2);
    let (sr1, dur1) = probe_samplerate_duration(p1).await?;
    let (sr2, dur2) = probe_samplerate_duration(p2).await?;

    // approximate samples count
    let samples1 = (sr1 as f64 * dur1).round() as usize;
    let samples2 = (sr2 as f64 * dur2).round() as usize;

    // choose shorter as reference
    let (ref_path, stream_path, _ref_samples_est, _sr_ref, _sr_stream) = if samples1 <= samples2 {
        (p1, p2, samples1, sr1, sr2)
    } else {
        (p2, p1, samples2, sr2, sr1)
    };

    // choose common sample rate (min)
    let target_sr = std::cmp::min(sr1, sr2);

    // no explicit pan expression here (ffmpeg will mix to mono with -ac 1)
    let pan_expr: Option<String> = None;

    // read reference fully in f32 (mono, resampled if needed)
    let (_sr_used, mut ref_samples) =
        read_all_pcm_f32_from_ffmpeg(ref_path, pan_expr.clone(), Some(target_sr)).await?;

    if ref_samples.is_empty() {
        anyhow::bail!("reference samples empty");
    }

    // reverse reference for convolution
    ref_samples.reverse();
    let m = ref_samples.len();

    // Safety: if reference is too large for our FFT cap, bail early with a clear error
    if m > MAX_N_CAP {
        anyhow::bail!(
            "reference is too large for streaming FFT approach (samples = {}, MAX_N_CAP = {}). \
             Consider increasing MAX_N_CAP or using a segmentation strategy.",
            m,
            MAX_N_CAP
        );
    }

    // block size B
    let b = BLOCK_SIZE_SAMPLES;
    let mut n = next_pow2(b + m - 1);
    if n > MAX_N_CAP {
        // if cap is smaller than required, reduce to cap (safe because m <= MAX_N_CAP)
        n = MAX_N_CAP;
    }
    // ensure n >= m
    if n < m {
        n = next_pow2(m);
        if n > MAX_N_CAP {
            anyhow::bail!(
                "computed FFT size exceeds MAX_N_CAP after adjusting to reference size (m = {}, needed n = {}, MAX_N_CAP = {})",
                m,
                n,
                MAX_N_CAP
            );
        }
    }

    // FFT planner (f32)
    let mut planner = FftPlanner::<f32>::new();

    // Precompute ref FFT for fixed n (if n >= m)
    let fft = planner.plan_fft_forward(n);
    let mut ref_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..m.min(n) {
        ref_buf[i] = Complex32::new(ref_samples[i], 0.0);
    }
    fft.process(&mut ref_buf);
    let ref_fft_pre = ref_buf; // keep

    // spawn ffmpeg for streaming file and keep child alive
    let mut child = spawn_ffmpeg_child(stream_path, pan_expr.clone(), Some(target_sr))?;
    let stdout = child.stdout.take().context("no stdout from streaming ffmpeg")?;
    let mut reader = tokio::io::BufReader::new(stdout);

    // prepare overlap and buffers (f32)
    let mut overlap: Vec<f32> = vec![0.0f32; if m >= 1 { m - 1 } else { 0 }];
    let block_bytes = b * 4;
    let mut local_buf: Vec<u8> = vec![0u8; block_bytes];

    // track max
    let mut maxv = f32::NEG_INFINITY;
    let mut max_idx_samples: usize = 0usize;
    let mut global_pos: usize = 0usize; // position in stream (sample index at start of this block)

    // pool (not heavily used here, but kept for compatibility)
    let _pool = Arc::new(Semaphore::new(pool_capacity));

    loop {
        let nread = reader.read(&mut local_buf).await?;
        if nread == 0 {
            break;
        }
        let samples_read = nread / 4;
        let mut block_samples: Vec<f32> = Vec::with_capacity(samples_read);
        for i in 0..samples_read {
            let base = i * 4;
            let bytes = [local_buf[base], local_buf[base + 1], local_buf[base + 2], local_buf[base + 3]];
            block_samples.push(f32::from_le_bytes(bytes));
        }

        // build input = overlap + block
        let l = overlap.len() + block_samples.len();
        let mut n_iter = next_pow2(l + m - 1);
        if n_iter > MAX_N_CAP {
            n_iter = MAX_N_CAP;
        }
        // ensure n_iter >= m (otherwise we'd index ref_samples out of bounds later)
        if n_iter < m {
            n_iter = next_pow2(m);
            if n_iter > MAX_N_CAP {
                anyhow::bail!(
                    "iteration FFT size exceeds MAX_N_CAP (m = {}, n_iter = {}, MAX_N_CAP = {})",
                    m,
                    n_iter,
                    MAX_N_CAP
                );
            }
        }

        // If n_iter equals precomputed n, reuse ref_fft_pre; otherwise compute ref_fft_local
        let mut ref_fft_local: Vec<Complex32>;
        if n_iter == n {
            ref_fft_local = ref_fft_pre.clone();
        } else {
            let fft_local = planner.plan_fft_forward(n_iter);
            ref_fft_local = vec![Complex32::new(0.0, 0.0); n_iter];
            let copy_n = std::cmp::min(m, n_iter);
            for i in 0..copy_n {
                ref_fft_local[i] = Complex32::new(ref_samples[i], 0.0);
            }
            fft_local.process(&mut ref_fft_local);
        }

        // prepare input buffer of size n_iter
        let mut in_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n_iter];
        // copy overlap
        for (i, &v) in overlap.iter().enumerate() {
            in_buf[i] = Complex32::new(v, 0.0);
        }
        // copy block
        for (i, &v) in block_samples.iter().enumerate() {
            let idx = overlap.len() + i;
            if idx < in_buf.len() {
                in_buf[idx] = Complex32::new(v, 0.0);
            } else {
                // should not happen because we sized n_iter >= l + m -1, but guard anyway
                break;
            }
        }

        // FFT input (create plan for n_iter)
        let fft_local = planner.plan_fft_forward(n_iter);
        let ifft_local = planner.plan_fft_inverse(n_iter);
        fft_local.process(&mut in_buf);

        // multiply by ref_fft_local (both length n_iter)
        for i in 0..n_iter {
            let a = in_buf[i];
            let b = ref_fft_local[i];
            in_buf[i] = a * b;
        }

        // inverse
        ifft_local.process(&mut in_buf);

        // valid output region: indices [m-1 .. m-1 + block_len -1]
        let start_idx = m.saturating_sub(1);
        let block_len = block_samples.len();
        for i in 0..block_len {
            let idx = start_idx + i;
            if idx >= in_buf.len() {
                break;
            }
            // rustfft inverse is NOT normalized, divide by n_iter
            let val = in_buf[idx].re / (n_iter as f32);
            let mag = val.abs();
            if mag > maxv {
                maxv = mag;
                max_idx_samples = global_pos + i;
            }
        }

        // update overlap to last m-1 samples of overlap+block
        let mut tail_source: Vec<f32> = Vec::with_capacity(overlap.len() + block_samples.len());
        tail_source.extend_from_slice(&overlap);
        tail_source.extend_from_slice(&block_samples);
        let overlap_len = if m >= 1 { m - 1 } else { 0 };
        if tail_source.len() >= overlap_len {
            let start = tail_source.len().saturating_sub(overlap_len);
            overlap.clear();
            overlap.extend_from_slice(&tail_source[start..]);
        } else {
            // pad left with zeros
            let mut new_overlap = vec![0.0f32; overlap_len];
            let pad = new_overlap.len().saturating_sub(tail_source.len());
            for i in 0..tail_source.len() {
                new_overlap[pad + i] = tail_source[i];
            }
            overlap = new_overlap;
        }

        global_pos += block_len;
    }

    // ensure child finishes
    let _ = child.wait().await;

    // decide which file to cut: if ref_path==p1 then cut in2 else cut in1
    let file_to_cut = if ref_path == p1 {
        in2.to_string()
    } else {
        in1.to_string()
    };

    // offset in seconds
    let offset_seconds = (max_idx_samples as f64) / (target_sr as f64);

    Ok(CorrelationResult { file: file_to_cut, offset_seconds })
}

/// Wrapper compatible with your main.rs
pub async fn second_correlation_async(in1: &str, in2: &str, pool_capacity: usize) -> Result<CorrelationResult> {
    // use streaming low-ram implementation by default
    second_correlation_streaming(in1, in2, pool_capacity).await
}
