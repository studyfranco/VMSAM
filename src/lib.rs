use anyhow::{bail, Context, Result};
use realfft::RealFftPlanner;
use serde::Serialize;
use std::io::{BufReader, Read};
use std::process::{Command, Stdio};

#[derive(Debug, Serialize)]
pub struct CorrelationResult {
    pub file: String,
    pub offset_seconds: f64,
}

fn probe_samplerate_duration(path: &str) -> Result<(u32, f64)> {
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
            path,
        ])
        .output()
        .with_context(|| format!("ffprobe failed for {path}"))?;
    if !out.status.success() {
        bail!("ffprobe returned non-zero for {path}");
    }
    let txt = String::from_utf8_lossy(&out.stdout);
    let mut sr: Option<u32> = None;
    let mut dur: Option<f64> = None;
    for line in txt.lines().map(str::trim).filter(|s| !s.is_empty()) {
        if sr.is_none() && line.chars().all(|c| c.is_ascii_digit()) {
            sr = line.parse::<u32>().ok();
        } else if dur.is_none() {
            dur = line.parse::<f64>().ok();
        }
    }
    Ok((sr.unwrap_or(44100), dur.unwrap_or(0.0)))
}

/// Mono f32 samples of the first audio stream at `sample_rate`, read from the ffmpeg pipe as they are decoded.
fn read_pcm_f32(path: &str, sample_rate: u32, duration: f64) -> Result<Vec<f32>> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-probesize",
            "1000M",
            "-threads",
            "3",
            "-vn",
            "-nostdin",
            "-i",
            path,
        ])
        .args(["-ac", "1", "-ar", &sample_rate.to_string()])
        .args(["-f", "f32le", "-acodec", "pcm_f32le", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("spawn ffmpeg")?;
    let mut reader = BufReader::with_capacity(
        1 << 20,
        child.stdout.take().context("no stdout from ffmpeg")?,
    );
    let mut samples =
        Vec::with_capacity((duration * sample_rate as f64) as usize + sample_rate as usize);
    let mut chunk = vec![0u8; 1 << 16];
    let mut pending = 0usize;
    loop {
        let nread = reader
            .read(&mut chunk[pending..])
            .context("read ffmpeg stdout")?;
        if nread == 0 {
            break;
        }
        let available = pending + nread;
        let whole = available - available % 4;
        samples.extend(
            chunk[..whole]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
        chunk.copy_within(whole..available, 0);
        pending = available - whole;
    }
    child.wait().context("wait ffmpeg")?;
    if samples.is_empty() {
        bail!("no audio decoded from {path}");
    }
    Ok(samples)
}

/// Full-range cross-correlation of the two files by real FFT; returns the file to cut and by how many seconds.
pub fn second_correlation(in1: &str, in2: &str) -> Result<CorrelationResult> {
    let (sr1, dur1) = probe_samplerate_duration(in1)?;
    let (sr2, dur2) = probe_samplerate_duration(in2)?;
    let target_sr = sr1.min(sr2);
    let mut s1 = read_pcm_f32(in1, target_sr, dur1)?;
    let mut s2 = read_pcm_f32(in2, target_sr, dur2)?;
    let (ls1, ls2) = (s1.len(), s2.len());

    // Smallest even 2^a 3^b 5^c length holding every lag: the next power of two can double the buffers.
    let needed = ls1 + ls2 - 1;
    let mut padsize = needed + needed % 2;
    loop {
        let mut rest = padsize;
        for factor in [2, 3, 5] {
            while rest % factor == 0 {
                rest /= factor;
            }
        }
        if rest == 1 {
            break;
        }
        padsize += 2;
    }

    // Each plan holds about as much memory as one spectrum: the inverse is planned once the forward is dropped.
    let forward = RealFftPlanner::<f32>::new().plan_fft_forward(padsize);
    s1.resize(padsize, 0.0);
    let mut spectrum = forward.make_output_vec();
    forward.process(&mut s1, &mut spectrum)?;
    drop(s1);
    s2.resize(padsize, 0.0);
    let mut spectrum2 = forward.make_output_vec();
    forward.process(&mut s2, &mut spectrum2)?;
    drop(forward);
    for (a, b) in spectrum.iter_mut().zip(spectrum2.iter()) {
        *a *= b.conj();
    }
    drop(spectrum2);
    // realfft returns an error when the DC or Nyquist bin keeps a rounding imaginary part.
    let last = spectrum.len() - 1;
    spectrum[0].im = 0.0;
    spectrum[last].im = 0.0;
    let correlation = &mut s2;
    RealFftPlanner::<f32>::new()
        .plan_fft_inverse(padsize)
        .process(&mut spectrum, correlation)?;

    // Index k < ls1 is file 1 ahead by k samples; padsize - k is file 2 ahead (the gap between them is padding).
    let mut xmax = 0usize;
    let mut maxv = f32::NEG_INFINITY;
    for k in (0..ls1).chain(padsize - ls2 + 1..padsize) {
        let mag = correlation[k].abs();
        if mag > maxv {
            maxv = mag;
            xmax = k;
        }
    }
    let fs = target_sr as f64;
    let result = if xmax < ls1 {
        CorrelationResult {
            file: in1.to_string(),
            offset_seconds: xmax as f64 / fs,
        }
    } else {
        CorrelationResult {
            file: in2.to_string(),
            offset_seconds: (padsize - xmax) as f64 / fs,
        }
    };
    Ok(result)
}
