use anyhow::{Context, Result};
use hound;
use rustfft::{FftPlanner, num_complex::Complex as C64};
use serde::Serialize;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::Semaphore;
use tokio::task;
use uuid::Uuid;

/// Résultat sérialisable
#[derive(Debug, Serialize)]
pub struct CorrelationResult {
    pub file: String,
    pub offset_seconds: f64,
}

/// Lecture WAV mono synchrones (pour spawn_blocking)
fn read_wav_mono_sync(path: &str) -> Result<(u32, Vec<f64>)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("opening wav file: {}", path))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;

    // lire tous les samples i16
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .map(|s| s.expect("wav read error"))
        .collect();

    // si mono -> renvoie directement
    if channels == 1 {
        let out = samples.into_iter().map(|s| s as f64).collect();
        return Ok((sr, out));
    }

    // si multi -> mix average
    let frames = samples.len() / channels;
    let mut out: Vec<f64> = Vec::with_capacity(frames);
    for frame_idx in 0..frames {
        let mut sum = 0i64;
        for ch in 0..channels {
            let idx = frame_idx * channels + ch;
            sum += samples[idx] as i64;
        }
        let avg = sum as f64 / channels as f64;
        out.push(avg);
    }
    Ok((sr, out))
}

async fn read_wav_mono(path: &str) -> Result<(u32, Vec<f64>)> {
    let p = path.to_string();
    task::spawn_blocking(move || read_wav_mono_sync(&p))
        .await
        .context("task join error")?
}

/// Exécute une commande ffmpeg (args) contrôlée par le pool (Semaphore)
async fn run_ffmpeg_job(pool: Arc<Semaphore>, args: Vec<String>) -> Result<()> {
    let permit = pool.acquire_owned().await.expect("semaphore closed");

    let mut cmd = Command::new("ffmpeg");
    cmd.args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let status = cmd
        .status()
        .await
        .with_context(|| format!("failed to spawn ffmpeg with args: {:?}", args))?;

    drop(permit);

    if !status.success() {
        anyhow::bail!("ffmpeg failed with status: {:?}", status);
    }
    Ok(())
}

/// appelle ffprobe pour obtenir le nombre de channels (async)
async fn probe_channel_count(path: &Path) -> Result<usize> {
    // ffprobe -v error -select_streams a:0 -show_entries stream=channels -of default=noprint_wrappers=1:nokey=1 infile
    let out = Command::new("ffprobe")
        .args(&[
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.to_string_lossy().as_ref(),
        ])
        .output()
        .await
        .with_context(|| format!("ffprobe failed for {:?}", path))?;

    if !out.status.success() {
        anyhow::bail!(
            "ffprobe returned non-zero for {:?} (stderr hidden)",
            path
        );
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let n: usize = s.parse().unwrap_or(1);
    Ok(n.max(1))
}

/// construit l'argument -af pour pan mix -> mono avec coefficients 1/N
fn build_pan_mono_expression(channels: usize) -> String {
    if channels <= 1 {
        return "pan=mono|c0=c0".to_string();
    }
    let coef = 1.0 / channels as f64;
    let mut parts = Vec::with_capacity(channels);
    for i in 0..channels {
        parts.push(format!("{coef:.6}*c{i}", coef = coef, i = i));
    }
    format!("pan=mono|c0={}", parts.join("+"))
}

/// génère les args ffmpeg pour normaliser et mixer en mono (pan then loudnorm)
async fn generate_norm_args_mix_mono(in_path: &Path, out_path: &Path) -> Result<Vec<String>> {
    let channels = probe_channel_count(in_path).await.unwrap_or(1);
    let pan = build_pan_mono_expression(channels);
    let filter = format!(
        "{},loudnorm=i=-23.0:lra=7.0:tp=-2.0:offset=4.45:linear=true:print_format=json",
        pan
    );

    Ok(vec![
        "-y".into(),
        "-threads".into(),
        "2".into(),
        "-nostdin".into(),
        "-i".into(),
        in_path.to_string_lossy().into_owned(),
        "-af".into(),
        filter,
        "-c:a".into(),
        "pcm_s16le".into(),
        out_path.to_string_lossy().into_owned(),
    ])
}

/// génère args pour denoise (afftdn)
fn generate_denoise_args(in_path: &Path, out_path: &Path) -> Vec<String> {
    vec![
        "-y".into(),
        "-threads".into(),
        "2".into(),
        "-nostdin".into(),
        "-i".into(),
        in_path.to_string_lossy().into_owned(),
        "-af".into(),
        "afftdn=nf=-25".into(),
        out_path.to_string_lossy().into_owned(),
    ]
}

/// read_normalized async : normalise + mix -> mono + (évent. denoise if still diff sample rate)
/// Retourne (fs, s1_mono, s2_mono)
pub async fn read_normalized_async(in1: &str, in2: &str, pool_capacity: usize) -> Result<(u32, Vec<f64>, Vec<f64>)> {
    let pool = Arc::new(Semaphore::new(pool_capacity));

    // Tentative de lecture initiale --- utile si input déjà WAV pcm -> on peut extraire sample rate
    let mut r1_opt = None;
    let mut r2_opt = None;
    if let Ok((sr1, _)) = task::spawn_blocking({
        let p = in1.to_string();
        move || read_wav_mono_sync(&p)
    }).await.context("join error")? {
        r1_opt = Some(sr1);
    }
    if let Ok((sr2, _)) = task::spawn_blocking({
        let p = in2.to_string();
        move || read_wav_mono_sync(&p)
    }).await.context("join error")? {
        r2_opt = Some(sr2);
    }

    // Si les deux taux existent et sont égaux -> lecture directe
    if let (Some(sr1), Some(sr2)) = (r1_opt, r2_opt) {
        if sr1 == sr2 {
            let (_rr1, ss1) = read_wav_mono(in1).await?;
            let (rr2, ss2) = read_wav_mono(in2).await?;
            // garantir cohérence
            if sr1 != rr2 {
                anyhow::bail!("sample rates mismatch on direct read: {} vs {}", sr1, rr2);
            }
            return Ok((sr1, ss1, ss2));
        }
    }

    // Sinon: normalisation vers fichiers temporaires mono
    let base1 = Path::new(in1)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("in1");
    let base2 = Path::new(in2)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("in2");
    let uid1 = Uuid::new_v4().to_string();
    let uid2 = Uuid::new_v4().to_string();

    let tmp_dir = std::env::temp_dir();
    let out1 = tmp_dir.join(format!("{}_{}_norm.wav", base1, uid1));
    let out2 = tmp_dir.join(format!("{}_{}_norm.wav", base2, uid2));

    // build args (async because we probe channels)
    let args1 = generate_norm_args_mix_mono(Path::new(in1), &out1).await?;
    let args2 = generate_norm_args_mix_mono(Path::new(in2), &out2).await?;

    // spawn both jobs under pool
    let p1 = {
        let pool = pool.clone();
        let a = args1.clone();
        tokio::spawn(async move { run_ffmpeg_job(pool, a).await })
    };
    let p2 = {
        let pool = pool.clone();
        let a = args2.clone();
        tokio::spawn(async move { run_ffmpeg_job(pool, a).await })
    };

    // await both
    let _ = p1.await.context("join norm job1")??;
    let _ = p2.await.context("join norm job2")??;

    // read resulting WAVs
    let (mut r1, mut s1) = read_wav_mono(out1.to_string_lossy().as_ref()).await?;
    let (mut r2, mut s2) = read_wav_mono(out2.to_string_lossy().as_ref()).await?;

    // if sample rates still different -> attempt denoise and reread
    if r1 != r2 {
        let out1_d = tmp_dir.join(format!("{}_{}_norm_denoise.wav", base1, uid1));
        let out2_d = tmp_dir.join(format!("{}_{}_norm_denoise.wav", base2, uid2));

        let args1 = generate_denoise_args(&out1, &out1_d);
        let args2 = generate_denoise_args(&out2, &out2_d);

        let p1 = {
            let pool = pool.clone();
            tokio::spawn(async move { run_ffmpeg_job(pool, args1).await })
        };
        let p2 = {
            let pool = pool.clone();
            tokio::spawn(async move { run_ffmpeg_job(pool, args2).await })
        };

        let _ = p1.await.context("join denoise job1")??;
        let _ = p2.await.context("join denoise job2")??;

        let (rr1, ss1) = read_wav_mono(out1_d.to_string_lossy().as_ref()).await?;
        let (rr2, ss2) = read_wav_mono(out2_d.to_string_lossy().as_ref()).await?;
        r1 = rr1; r2 = rr2; s1 = ss1; s2 = ss2;

        // cleanup both norm and denoise files
        let _ = tokio::fs::remove_file(out1).await;
        let _ = tokio::fs::remove_file(out2).await;
        let _ = tokio::fs::remove_file(out1_d).await;
        let _ = tokio::fs::remove_file(out2_d).await;
    } else {
        // cleanup norm files
        let _ = tokio::fs::remove_file(out1).await;
        let _ = tokio::fs::remove_file(out2).await;
    }

    if r1 != r2 {
        anyhow::bail!("not same sample rate after normalization attempts: {} vs {}", r1, r2);
    }

    Ok((r1, s1, s2))
}

/// corrabs sync (FFT) — spawn_blocking caller
fn corrabs_sync(s1: &[f64], s2: &[f64]) -> Result<(usize, usize)> {
    let ls1 = s1.len();
    let ls2 = s2.len();
    // taille de convolution: ls1 + ls2 - 1 (classique); on prend puissance de 2 >=
    let needed = ls1 + ls2 - 1;
    let mut padsize = 1usize;
    while padsize < needed {
        padsize <<= 1;
    }

    // allouer vecteurs complexes f64
    let mut a: Vec<C64<f64>> = vec![C64::new(0.0, 0.0); padsize];
    let mut b: Vec<C64<f64>> = vec![C64::new(0.0, 0.0); padsize];

    // copier signaux réels dans la partie réelle
    for i in 0..ls1 {
        a[i] = C64::new(s1[i], 0.0);
    }
    for i in 0..ls2 {
        b[i] = C64::new(s2[i], 0.0);
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(padsize);
    let ifft = planner.plan_fft_inverse(padsize);

    fft.process(&mut a);
    fft.process(&mut b);

    // produit a * conj(b)
    let mut prod: Vec<C64<f64>> = vec![C64::new(0.0, 0.0); padsize];
    for i in 0..padsize {
        let conj_b = C64::new(b[i].re, -b[i].im);
        prod[i] = a[i] * conj_b;
    }

    ifft.process(&mut prod);

    // magnitude max
    let mut xmax = 0usize;
    let mut maxv = f64::NEG_INFINITY;
    for (i, val) in prod.iter().enumerate() {
        let mag = (val.re.powi(2) + val.im.powi(2)).sqrt();
        if mag > maxv {
            maxv = mag;
            xmax = i;
        }
    }
    Ok((padsize, xmax))
}

async fn corrabs(s1: Vec<f64>, s2: Vec<f64>) -> Result<(usize, usize)> {
    task::spawn_blocking(move || corrabs_sync(&s1, &s2))
        .await
        .context("corrabs join error")?
}

/// second_correlation async principal
pub async fn second_correlation_async(in1: &str, in2: &str, pool_capacity: usize) -> Result<CorrelationResult> {
    let (fs, s1, s2) = read_normalized_async(in1, in2, pool_capacity).await?;
    let (padsize, xmax) = corrabs(s1, s2).await?;

    let fs_f = fs as f64;
    // interprétation lag: index > padsize/2 => lag négatif (s2 en avance), sinon lag positif (s1 en avance)
    let (file_to_cut, offset_seconds) = if xmax > padsize / 2 {
        (in2.to_string(), (padsize - xmax) as f64 / fs_f)
    } else {
        (in1.to_string(), (xmax) as f64 / fs_f)
    };

    Ok(CorrelationResult { file: file_to_cut, offset_seconds })
}
