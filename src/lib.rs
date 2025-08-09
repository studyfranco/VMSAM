// Assurez-vous d'avoir ces dépendances dans Cargo.toml:
// tokio = { version = "1.37", features = ["full"] }
// rustfft = "6.0"
// num-complex = "0.4"
// anyhow = "1.0"

use anyhow::{Context, Result};
use num_complex::Complex32;
use rustfft::{FftPlanner};
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::Semaphore;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct CorrelationResult {
    pub file: String,
    pub offset_seconds: f64,
}

/// Paramètres (à ajuster)
const BLOCK_SIZE_SAMPLES: usize = 131_072; // ~0.5 MB of f32 (131k * 4 = 524kB)
const MAX_FFMPEG_POOL: usize = 2;

/// Probe sample rate and duration (seconds) using ffprobe.
/// Returns (sample_rate, duration_seconds)
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
    // ffprobe returns lines: sample_rate\n duration\n  (order may vary)
    let mut sr: Option<u32> = None;
    let mut dur: Option<f64> = None;
    for line in txt.lines() {
        let s = line.trim();
        if s.is_empty() { continue; }
        if sr.is_none() && s.chars().all(|c| c.is_digit(10)) {
            if let Ok(v) = s.parse::<u32>() { sr = Some(v); continue; }
        }
        if dur.is_none() {
            if let Ok(v) = s.parse::<f64>() { dur = Some(v); continue; }
        }
    }
    let sr = sr.unwrap_or(48000);
    let dur = dur.unwrap_or(0.0);
    Ok((sr, dur))
}

/// Spawn ffmpeg to output mono pcm_f32le to stdout.
/// Optionally resample via `target_sr` (None => leave sample rate).
fn spawn_ffmpeg_pipe(path: &Path, pan_expr: Option<String>, target_sr: Option<u32>) -> Result<tokio::process::ChildStdout> {
    // build args:
    // -i in -af "<pan_expr>" -ac 1 [-ar target_sr] -f f32le -acodec pcm_f32le -
    let mut cmd = Command::new("ffmpeg");
    let mut args = vec![
        "-vn".to_string(),
        "-nostdin".to_string(),
        "-i".to_string(),
        path.to_string_lossy().into_owned(),
    ];
    if let Some(pan) = pan_expr {
        args.push("-af".to_string());
        args.push(pan);
    } else {
        // force mono simple
        // we'll rely on -ac 1 below
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

    cmd.args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);

    let mut child = cmd.spawn().context("failed to spawn ffmpeg")?;
    let stdout = child.stdout.take().context("no stdout from ffmpeg")?;
    Ok(stdout)
}

/// Read the whole stdin (pcm_f32le) into a Vec<f32> (for the reference file).
/// We deliberately use f32 to reduce RAM.
async fn read_all_pcm_f32_from_ffmpeg(path: &Path, pan_expr: Option<String>, target_sr: Option<u32>) -> Result<(u32, Vec<f32>)> {
    // probe sr first
    let (sr0, _dur) = probe_samplerate_duration(path).await?;
    let target_sr = target_sr.unwrap_or(sr0);

    let mut out = Command::new("ffmpeg")
        .args(&[
            "-vn", "-nostdin",
            "-i", path.to_string_lossy().as_ref(),
        ])
        .args(if let Some(p) = pan_expr {
            vec!["-af", &p, "-ac", "1"]
        } else {
            vec!["-ac", "1"]
        })
        .args(&["-ar", &format!("{}", target_sr), "-f", "f32le", "-acodec", "pcm_f32le", "-"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("spawn ffmpeg for read_all")?
        .stdout
        .context("no stdout")?;

    let mut reader = tokio::io::BufReader::new(out);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).await.context("read ffmpeg stdout")?;

    // convert bytes -> f32 little endian
    let mut samples = Vec::with_capacity(buf.len() / 4);
    for chunk in buf.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        samples.push(f32::from_le_bytes(bytes));
    }
    Ok((target_sr, samples))
}

/// Core: streaming cross-correlation using overlap-save:
/// - load the smaller file fully as reference (rev)
/// - stream the larger file in blocks, compute convolution with reversed ref via FFT
pub async fn second_correlation_streaming(in1: &str, in2: &str, pool_capacity: usize) -> Result<CorrelationResult> {
    // probe both files (sample rate + duration)
    let p1 = Path::new(in1);
    let p2 = Path::new(in2);
    let (sr1, dur1) = probe_samplerate_duration(p1).await?;
    let (sr2, dur2) = probe_samplerate_duration(p2).await?;

    // compute approximate samples
    let samples1 = (sr1 as f64 * dur1).round() as usize;
    let samples2 = (sr2 as f64 * dur2).round() as usize;

    // choose shorter as reference
    let (ref_path, stream_path, ref_samples_est, sr_ref, sr_stream) = if samples1 <= samples2 {
        (p1, p2, samples1, sr1, sr2)
    } else {
        (p2, p1, samples2, sr2, sr1)
    };

    // choose common sample rate (min)
    let target_sr = std::cmp::min(sr_ref, sr_stream);

    // optional pan expression: simple average mix (1/N)
    // for robustness we leave pan None and rely on -ac 1 (ffmpeg do mixing).
    let pan_expr: Option<String> = None;

    // read reference fully in f32 (mono, resampled if needed)
    let (_sr_ref_used, mut ref_samples) =
        read_all_pcm_f32_from_ffmpeg(ref_path, pan_expr.clone(), Some(target_sr)).await?;

    if ref_samples.is_empty() {
        anyhow::bail!("reference samples empty");
    }

    // reverse reference for correlation (we will convolve streaming signal with reversed reference)
    ref_samples.reverse();
    let m = ref_samples.len();

    // Prepare FFT planner and reference FFT for chosen N (we'll decide N per block)
    // We'll pick a block size B (samples per iteration), and choose N = next_pow2(B + m - 1).
    let b = BLOCK_SIZE_SAMPLES;
    let n_min = next_pow2(b + m - 1);

    // but if m is big, n_min may be huge; ensure N not excessive:
    let mut n = n_min;
    // optional cap N size: e.g. don't exceed 1<<22 (~4M) to bound mem; adjust as needed
    let max_n = 1 << 22;
    if n > max_n {
        n = max_n;
    }

    // compute FFT of ref padded to N (Complex32)
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // prepare ref_fft
    let mut ref_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..m.min(n) {
        ref_buf[i] = Complex32::new(ref_samples[i], 0.0);
    }
    fft.process(&mut ref_buf);
    let ref_fft = ref_buf.clone(); // store for reuse

    // spawn ffmpeg on streaming file and read blocks from stdout
    let mut stdout = spawn_ffmpeg_pipe(stream_path, pan_expr.clone(), Some(target_sr))?;
    let mut reader = tokio::io::BufReader::new(stdout);

    // we'll read bytes for one block: block bytes = B * 4 (f32)
    let block_bytes = b * 4;
    let mut overlap: Vec<f32> = vec![0.0f32; m - 1]; // M-1 overlap initially zero
    let mut local_buf: Vec<u8> = vec![0u8; block_bytes];

    // track max
    let mut maxv = f32::NEG_INFINITY;
    let mut max_idx_samples: usize = 0usize;
    let mut global_pos: usize = 0usize; // position in stream (sample index at start of this block)

    // semaphore only used if we spawn ffmpeg jobs parallel elsewhere; keep for compatibility
    let _pool = Arc::new(Semaphore::new(pool_capacity));

    loop {
        // read up to block_bytes from ffmpeg stdout
        let nread = reader.read(&mut local_buf).await?;
        if nread == 0 { break; } // EOF

        // if nread not multiple of 4, trim
        let samples_read = nread / 4;
        let mut block_samples: Vec<f32> = Vec::with_capacity(samples_read);
        for i in 0..samples_read {
            let base = i * 4;
            let bytes = [local_buf[base], local_buf[base+1], local_buf[base+2], local_buf[base+3]];
            block_samples.push(f32::from_le_bytes(bytes));
        }

        // Build input segment = overlap (m-1) + block_samples (B' maybe smaller)
        let l = overlap.len() + block_samples.len();
        // Choose N for this iteration: N >= l + m -1  (convolution length)
        let mut N = next_pow2(l + m - 1);
        if N > max_n { N = max_n; } // cap

        // prepare buffers (complex)
        let mut in_buf: Vec<Complex32> = vec![Complex32::new(0.0,0.0); N];
        // copy overlap
        for (i, &v) in overlap.iter().enumerate() {
            in_buf[i] = Complex32::new(v, 0.0);
        }
        // copy block
        for (i, &v) in block_samples.iter().enumerate() {
            in_buf[overlap.len() + i] = Complex32::new(v, 0.0);
        }
        // zero padding already present

        // compute FFT of input (if N differs from ref_fft size, we need ref FFT resized)
        // We'll compute ref_fft for this N on the fly if sizes differ (rare if we fix N).
        let fft_local = planner.plan_fft_forward(N);
        let ifft_local = planner.plan_fft_inverse(N);

        // prepare ref_fft_local
        let mut ref_fft_local: Vec<Complex32> = vec![Complex32::new(0.0,0.0); N];
        // place reversed ref (we previously reversed ref_samples) into beginning
        let copy_n = std::cmp::min(m, N);
        for i in 0..copy_n {
            ref_fft_local[i] = Complex32::new(ref_samples[i], 0.0);
        }
        fft_local.process(&mut ref_fft_local);

        // FFT input
        fft_local.process(&mut in_buf);

        // pointwise multiply in_buf * ref_fft_local (complex multiply)
        for i in 0..N {
            let a = in_buf[i];
            let b = ref_fft_local[i];
            // a *= b
            in_buf[i] = a * b;
        }

        // ifft
        ifft_local.process(&mut in_buf);

        // valid output region: indices [m-1 .. m-1 + block_len - 1] (length block_len)
        let start_idx = m.saturating_sub(1);
        let block_len = block_samples.len();
        for i in 0..block_len {
            let idx = start_idx + i;
            if idx >= in_buf.len() { break; }
            // The convolution result is in_buf[idx].re / N if the ifft is unnormalized in rustfft. rustfft does NOT normalize IFFT (it scales by 1/N if using it?), actually rustfft's inverse does not scale — you must divide by N.
            // rustfft's inverse does NOT scale, so divide by N:
            let val = in_buf[idx].re / (N as f32);
            let mag = val.abs();
            if mag > maxv {
                maxv = mag;
                max_idx_samples = global_pos + i;
            }
        }

        // update overlap to last m-1 samples of (overlap + block)
        // Build temp vec = overlap + block_samples, then take tail
        let mut tail_source_len = overlap.len() + block_samples.len();
        let mut tail_source: Vec<f32> = Vec::with_capacity(tail_source_len);
        tail_source.extend_from_slice(&overlap);
        tail_source.extend_from_slice(&block_samples);
        if tail_source.len() >= m - 1 {
            let start = tail_source.len() - (m - 1);
            overlap.clear();
            overlap.extend_from_slice(&tail_source[start..]);
        } else {
            // smaller than m-1: pad with zeros on the left
            let mut new_overlap = vec![0.0f32; m - 1 - tail_source.len()];
            new_overlap.extend_from_slice(&tail_source);
            overlap = new_overlap;
        }

        // advance global pos
        global_pos += block_len;
    } // end loop reading

    // Compute offset_seconds and file choice: if reference was in1 then file_to_cut=in2 else in1.
    // We chose ref_path earlier accordingly.
    let file_to_cut = if ref_path == p1 { in2.to_string() } else { in1.to_string() };
    // The correlation peak index indicates the best alignment: when ref was reversed, peak index corresponds to...
    // Interpretation: global sample index (max_idx_samples) is the lag where s_stream * s_ref (cross-correlation) achieves maximum.
    // offset_seconds = max_idx_samples / target_sr
    let offset_seconds = (max_idx_samples as f64) / (target_sr as f64);

    Ok(CorrelationResult { file: file_to_cut, offset_seconds })
}

/// next power of two
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

pub async fn second_correlation_async(in1: &str, in2: &str, pool_capacity: usize) -> anyhow::Result<CorrelationResult> {
    second_correlation_streaming(in1, in2, pool_capacity).await
}