# VMSAM — File Map

Where things live and what they do. Line counts are indicative; the write status
is authoritative and comes from `WRITE_ZONES.MD`.

| status | meaning |
| :--- | :--- |
| 🔒 | frozen — no modification |
| 🏷️ | frozen except inside `AGENT modification` tags |
| ✅ | open |

## Entry points

| file | l. | status | role |
| :--- | ---: | :---: | :--- |
| `main_gestionar_show.py` | 494 | 🔒 | series daemon: scan loop, and forks **both** uvicorn instances |
| `main.py` | 78 | 🔒 | CLI: merge an explicit list of files |
| `infinite_wrapper.py` | 143 | 🔒 | loop around `main.py` for `software=main` |
| `folder_generator.py` | 40 | ✅ | standalone folder scaffolding, no importer |

## Merge engine

| file | l. | status | role |
| :--- | ---: | :---: | :--- |
| `mergeVideo.py` | 2122 | 🏷️ | the heart. `merge_videos()` drives ranking, delay, assembly. Tags in `remove_not_compatible_video`, `adjust_delay_to_frame`, `keep_best_audio` |
| `video.py` | 1145 | 🏷️ | `video` class: probing, cutting, MD5, ffmpeg pools. Tags in `get_best_quality_video`, `get_good_frame` |
| `audioCorrelation.py` | 309 | 🔒 | stages 1–2: fingerprint and FFT correlation, Rust `audio_sync` behind it |
| `tools.py` | 394 | 🏷️ | shared runtime state, subprocess wrappers, episode lock, `tools.logs` |
| `merge_video_unused.py` | 79 | ✅ | extracted from `mergeVideo.py`, no importer |

## Dormant — no importer

| file | l. | status | state |
| :--- | ---: | :---: | :--- |
| `get_cut_time.py` | 565 | ✅ | multi-range cut detection, drift, audio chimera. Imports `frame_compare` |
| `frame_compare.py` | 221 | ✅ | pHash/DCT frame comparison. Compiles since campaign 2's repair |

## API — `gestionar_show`, entirely frozen

| file | l. | role |
| :--- | ---: | :--- |
| `api.py` | 382 | public app, `0.0.0.0:8080`, workers=5. Stateless, proxies fusion over loopback |
| `internal_api.py` | 108 | internal app, `127.0.0.1:42085`, **workers=1**. Owns the queue |
| `fusion.py` | 381 | the queue, its worker, and the repair job. Replaces library files |
| `internal_client.py` | 55 | stdlib urllib client, empty `ProxyHandler` |
| `model.py` | 295 | SQLAlchemy models and queries. **Schema is frozen** |
| `settings.py` | 13 | `DATABASE_URL`, shared by both instances |

`gestionar_movie/` (419 l.) is a complete app with no launcher in `run.sh`. Open.

## Rust

`lib.rs` — FFT cross-correlation core, the optimised rewrite that avoided the OOM.
`main.rs` — CLI `audio_sync <file1> <file2> [pool_size]`. Must pass `cargo check`,
`cargo clippy`, `cargo fmt`.

## Configuration

`config.ini` — `[software]` binary paths, `[mergerules]` codec ranking. **Additions
only.** `config.json`, `param_template.json`, `titles_subs_group.json` — frozen
unless the owner decides otherwise.

## Data flow

`run.sh` → `main_gestionar_show` → scan → regex → `process_episode` →
`mergeVideo.merge_videos` → library, **or** rejection → `incompatible_files` →
`POST /fusion` → internal worker → `fusion.run_fusion_job` → `merge_videos` again.

## The merge, stage by stage

The call tree is in `AGENT.MD` § Synchronization pipeline; this is what each step
actually does, read from the code.

**Preparation.** `prepare_get_delay_sub` picks the comparison audio parameters —
lowest channel count and sampling rate of the pair — then cuts `video.number_cut`
windows. Their length is **twice** their spacing, so consecutive windows overlap by
half and roughly 83 % of the episode is covered, every interior moment twice. Each
window is extracted per audio track by ffmpeg and loudness-normalised, so what is
compared differs only in timing and content.

**Stage 1 — `first_delay_test`.** `get_delay_fidelity` launches one thread per
*(track pair × window)* and each one runs `audioCorrelation.correlate`: Chromaprint
fingerprints, compared over a sliding span. It yields a delay and a fidelity per
window. The delays must agree across windows and across track pairs; those that
disagree are put in `ignore_audio_couple` and dropped from stage 2.

**Stage 2 — `second_delay_test`.** Re-extracts the audio as mono PCM around the
stage-1 answer, then `second_correlation` refines it — the Rust `audio_sync` binary,
with a Python fallback. It works on **one long block**, not on the stage-1 windows.
It runs unconditionally: it is the correction path for stage-1 fingerprint peaks,
not a second opinion sought when stage 1 looks doubtful.

**If Stage 1 fail The verdict — `test_if_constant_good_delay`.** `calculated_delay = stage 1 +
stage 2 × 1000`, then `abs(calculated_delay - stage 1) < 500`. That difference *is*
`stage 2 × 1000`, so the check is: **stage 2's correction must stay under 500 ms**.
Stage 2 works on a candidate already shifted by stage 1's answer, so it measures a
residual — a large one means stage 1 was wrong, and the pair is refused.

**Stage 3 — `get_best_video`, and only when no reference is imposed.** With a forced
reference the run takes the `else` branch of `compare_video.run`, which snaps the
delay and stops. Otherwise `video.get_good_frame` runs a **PSNR** scan over ±2
frames, and `video.get_best_quality_video` a **VMAF** vote to name the reference.

**Frame snap — `adjust_delay_to_frame`.** Called from `compare_video` only. On a CFR
reference the delay is rounded to a whole frame (41.708 ms at 23.976 fps); on a
non-CFR one it is returned unchanged, with a `TODO`. The final `mkvmerge --sync`
takes integer milliseconds, so the snapped value loses up to ~0.5 ms on the way
out — and `delay_same_md5_audio`, computed from container delays, is never snapped
at all.

**Stage 4 — `remove_not_compatible_video`.** Whatever raised in `compare_video.run`
was caught there and recorded as an incompatibility. This function drops the file
and, below two files, gives up on the merge. **It is where the repair happens** —
see `SPEC_ZONE_A.MD`.
