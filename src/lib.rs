use anyhow::{Context, Result};
use hound;
use rustfft::{FftPlanner, num_complex::Complex as C64};
use serde::Serialize;
use std::collections::VecDeque;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex, OnceLock};
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

/// Pool global de buffers FFT pour réduire les allocations
static BUFFER_POOL: OnceLock<Mutex<BufferPool>> = OnceLock::new();

struct BufferPool {
    buffers: VecDeque<Vec<C64<f64>>>,
    max_pool_size: usize,
}

impl BufferPool {
    fn new(max_size: usize) -> Self {
        Self {
            buffers: VecDeque::with_capacity(max_size),
            max_pool_size: max_size,
        }
    }

    fn get_buffer(&mut self, size: usize) -> Vec<C64<f64>> {
        // Chercher un buffer de taille suffisante
        if let Some(mut buffer) = self.buffers.pop_front() {
            if buffer.len() >= size {
                buffer.truncate(size);
                // Réinitialiser tous les éléments à zéro
                buffer.fill(C64::new(0.0, 0.0));
                return buffer;
            }
            // Si trop petit, le remettre dans le pool
            self.buffers.push_back(buffer);
        }
        
        // Allouer un nouveau buffer
        vec![C64::new(0.0, 0.0); size]
    }

    fn return_buffer(&mut self, buffer: Vec<C64<f64>>) {
        // Limiter la taille du pool pour éviter l'accumulation excessive
        if self.buffers.len() < self.max_pool_size && buffer.capacity() >= 1024 {
            self.buffers.push_back(buffer);
        }
    }
}

fn get_buffer_pool() -> &'static Mutex<BufferPool> {
    BUFFER_POOL.get_or_init(|| Mutex::new(BufferPool::new(8)))
}

/// Processeur de corrélation réutilisable pour éviter les allocations répétées
pub struct CorrelationProcessor {
    fft_planner: FftPlanner<f64>,
    scratch_buffer: Vec<C64<f64>>,
}

impl CorrelationProcessor {
    pub fn new() -> Self {
        Self {
            fft_planner: FftPlanner::new(),
            scratch_buffer: Vec::new(),
        }
    }

    /// Version optimisée de corrabs avec réutilisation des buffers
    pub fn correlate_optimized(&mut self, s1: &[f64], s2: &[f64]) -> Result<(usize, usize)> {
        let ls1 = s1.len();
        let ls2 = s2.len();
        let needed = ls1 + ls2 - 1;
        let mut padsize = 1usize;
        while padsize < needed {
            padsize <<= 1;
        }

        // Obtenir les buffers du pool
        let mut pool = get_buffer_pool().lock().unwrap();
        let mut buffer1 = pool.get_buffer(padsize);
        let mut buffer2 = pool.get_buffer(padsize);
        drop(pool); // Libérer le mutex rapidement

        // Préparer les FFT planners
        let fft = self.fft_planner.plan_fft_forward(padsize);
        let ifft = self.fft_planner.plan_fft_inverse(padsize);

        // Copier s1 dans buffer1
        for i in 0..ls1 {
            buffer1[i] = C64::new(s1[i], 0.0);
        }
        // Assurer que le reste est à zéro (déjà fait par get_buffer)

        // Copier s2 dans buffer2
        for i in 0..ls2 {
            buffer2[i] = C64::new(s2[i], 0.0);
        }

        // FFT des deux signaux
        fft.process(&mut buffer1);
        fft.process(&mut buffer2);

        // Produit conjugué in-place dans buffer1
        for i in 0..padsize {
            let conj_b = C64::new(buffer2[i].re, -buffer2[i].im);
            buffer1[i] *= conj_b;
        }

        // IFFT du résultat
        ifft.process(&mut buffer1);

        // Recherche du maximum
        let mut xmax = 0usize;
        let mut maxv = f64::NEG_INFINITY;
        for (i, val) in buffer1.iter().enumerate() {
            let mag = val.norm_sqr().sqrt(); // Plus efficace que re² + im²
            if mag > maxv {
                maxv = mag;
                xmax = i;
            }
        }

        // Retourner les buffers au pool
        let mut pool = get_buffer_pool().lock().unwrap();
        pool.return_buffer(buffer1);
        pool.return_buffer(buffer2);

        Ok((padsize, xmax))
    }
}

/// Lecture WAV mono synchrone optimisée
fn read_wav_mono_sync_optimized(path: &str) -> Result<(u32, Vec<f64>)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("opening wav file: {}", path))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;

    // Optimisation : éviter la double allocation pour le cas mono
    if channels == 1 {
        let total_samples = reader.len() as usize;
        let mut samples = Vec::with_capacity(total_samples);
        
        // Conversion directe avec normalisation
        for sample_result in reader.samples::<i16>() {
            let sample = sample_result.context("wav read error")?;
            samples.push(sample as f64 / 32768.0); // Normalisation directe
        }
        return Ok((sr, samples));
    }

    // Cas multi-canal : traitement par chunks pour réduire les pics mémoire
    let total_samples = reader.len() as usize;
    let frames = total_samples / channels;
    let mut mono_samples = Vec::with_capacity(frames);
    
    let chunk_size = 4096; // Traitement par chunks de 4096 échantillons
    let mut chunk = Vec::with_capacity(chunk_size * channels);
    
    for sample_result in reader.samples::<i16>() {
        let sample = sample_result.context("wav read error")?;
        chunk.push(sample);
        
        if chunk.len() == chunk_size * channels {
            // Traiter le chunk
            for frame_idx in 0..(chunk_size) {
                let mut sum = 0i64;
                for ch in 0..channels {
                    let idx = frame_idx * channels + ch;
                    if idx < chunk.len() {
                        sum += chunk[idx] as i64;
                    }
                }
                let avg = (sum as f64 / channels as f64) / 32768.0; // Normalisation
                mono_samples.push(avg);
            }
            chunk.clear();
        }
    }
    
    // Traiter le chunk restant
    if !chunk.is_empty() {
        let remaining_frames = chunk.len() / channels;
        for frame_idx in 0..remaining_frames {
            let mut sum = 0i64;
            for ch in 0..channels {
                let idx = frame_idx * channels + ch;
                sum += chunk[idx] as i64;
            }
            let avg = (sum as f64 / channels as f64) / 32768.0;
            mono_samples.push(avg);
        }
    }
    
    Ok((sr, mono_samples))
}

async fn read_wav_mono_optimized(path: &str) -> Result<(u32, Vec<f64>)> {
    let p = path.to_string();
    task::spawn_blocking(move || read_wav_mono_sync_optimized(&p))
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

/// read_normalized async optimisé : normalise + mix -> mono + (évent. denoise if still diff sample rate)
/// Retourne (fs, s1_mono, s2_mono)
pub async fn read_normalized_async_optimized(in1: &str, in2: &str, pool_capacity: usize) -> Result<(u32, Vec<f64>, Vec<f64>)> {
    let pool = Arc::new(Semaphore::new(pool_capacity));

    // Tentative de lecture initiale optimisée
    let mut r1_opt = None;
    let mut r2_opt = None;
    if let Ok((sr1, _)) = task::spawn_blocking({
        let p = in1.to_string();
        move || read_wav_mono_sync_optimized(&p)
    }).await.context("join error")? {
        r1_opt = Some(sr1);
    }
    if let Ok((sr2, _)) = task::spawn_blocking({
        let p = in2.to_string();
        move || read_wav_mono_sync_optimized(&p)
    }).await.context("join error")? {
        r2_opt = Some(sr2);
    }

    // Si les deux taux existent et sont égaux -> lecture directe optimisée
    if let (Some(sr1), Some(sr2)) = (r1_opt, r2_opt) {
        if sr1 == sr2 {
            let (_rr1, ss1) = read_wav_mono_optimized(in1).await?;
            let (rr2, ss2) = read_wav_mono_optimized(in2).await?;
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

    // read resulting WAVs avec lecture optimisée
    let (mut r1, mut s1) = read_wav_mono_optimized(out1.to_string_lossy().as_ref()).await?;
    let (mut r2, mut s2) = read_wav_mono_optimized(out2.to_string_lossy().as_ref()).await?;

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

        let (rr1, ss1) = read_wav_mono_optimized(out1_d.to_string_lossy().as_ref()).await?;
        let (rr2, ss2) = read_wav_mono_optimized(out2_d.to_string_lossy().as_ref()).await?;
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

async fn corrabs_optimized(s1: Vec<f64>, s2: Vec<f64>, processor: &mut CorrelationProcessor) -> Result<(usize, usize)> {
    let s1_clone = s1.clone();
    let s2_clone = s2.clone();
    
    task::spawn_blocking(move || {
        // Créer un processeur temporaire pour ce thread
        let mut temp_processor = CorrelationProcessor::new();
        temp_processor.correlate_optimized(&s1_clone, &s2_clone)
    })
    .await
    .context("corrabs join error")?
}

/// second_correlation async principal optimisé
pub async fn second_correlation_async_optimized(
    in1: &str, 
    in2: &str, 
    pool_capacity: usize
) -> Result<CorrelationResult> {
    let (fs, s1, s2) = read_normalized_async_optimized(in1, in2, pool_capacity).await?;
    
    // Créer un processeur pour cette corrélation
    let mut processor = CorrelationProcessor::new();
    let (padsize, xmax) = corrabs_optimized(s1, s2, &mut processor).await?;

    let fs_f = fs as f64;
    let (file_to_cut, offset_seconds) = if xmax > padsize / 2 {
        (in2.to_string(), (padsize - xmax) as f64 / fs_f)
    } else {
        (in1.to_string(), (xmax) as f64 / fs_f)
    };

    Ok(CorrelationResult { file: file_to_cut, offset_seconds })
}

/// Version avec processeur réutilisable pour traitement en batch
pub async fn second_correlation_with_processor(
    in1: &str, 
    in2: &str, 
    processor: &mut CorrelationProcessor,
    pool_capacity: usize
) -> Result<CorrelationResult> {
    let (fs, s1, s2) = read_normalized_async_optimized(in1, in2, pool_capacity).await?;
    let (padsize, xmax) = corrabs_optimized(s1, s2, processor).await?;

    let fs_f = fs as f64;
    let (file_to_cut, offset_seconds) = if xmax > padsize / 2 {
        (in2.to_string(), (padsize - xmax) as f64 / fs_f)
    } else {
        (in1.to_string(), (xmax) as f64 / fs_f)
    };

    Ok(CorrelationResult { file: file_to_cut, offset_seconds })
}

// Fonctions de compatibilité pour remplacer directement l'ancienne version
pub async fn read_normalized_async(in1: &str, in2: &str, pool_capacity: usize) -> Result<(u32, Vec<f64>, Vec<f64>)> {
    read_normalized_async_optimized(in1, in2, pool_capacity).await
}

pub async fn second_correlation_async(in1: &str, in2: &str, pool_capacity: usize) -> Result<CorrelationResult> {
    second_correlation_async_optimized(in1, in2, pool_capacity).await
}