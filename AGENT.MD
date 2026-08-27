# VMSAM — AI Agent Context & Directives

Hello AI Agent! You are assisting with the development of `vmsam` (Video & Audio Synchronization and Merging). Read and strictly adhere to these guidelines before inspecting, writing, or modifying any code.

---

## 📄 Document & Context Hierarchy

To keep this context clean and modular, responsibilities are split across dedicated files:
1. **`AGENT.MD`** *(This file)*: Core architecture rules, synchronization pipeline concepts, stage definitions, handled/unhandled cases, coding directives, security constraints, documentation standards, and testing directives.
2. **`FILE_MAP.MD`**: Detailed file listings, module responsibilities, and code indexing (maintained by the AI Agent).
3. **`AGENT_NOTES.MD`**: Agent worklog. You MUST update this file with architectural decisions, progress notes, or pending tasks after completing significant modifications.

---

## 🎯 Project Overview

`vmsam` is an automated media processing engine designed to analyze multiple video releases of the same title (from different sources, encodes, or languages), evaluate stream quality, align timelines frame-accurately, and assemble a single, optimal `.mkv` file containing the best video, audio, and subtitle streams available.

---

## 🔄 Synchronization & Matching Pipeline

The core engine operates through a multi-stage detection and alignment process to evaluate media compatibility and compute precise stream delays:

| Stage | Mechanism | Purpose & Operation |
| :--- | :--- | :--- |
| **Stage 1: Coarse Audio Fingerprinting** | Chromaprint (`fpcalc`) | Extracts short normalized audio cuts across the timeline and computes acoustic fingerprints to establish a coarse time offset. |
| **Stage 2: Fine Cross-Correlation** | FFT Signal Correlation | Executes full FFT signal correlation on raw PCM audio using a high-performance Rust binary (`audio_sync`) with a Python fallback for millisecond-accurate alignment. |
| **Stage 3: Frame-Accurate Alignment** | Visual PSNR Scanning | Evaluates visual similarity (PSNR) across a localized frame window around the audio offset to lock alignment to exact video frame boundaries. |
| **Stage 4: Incompatibility & Error Handling** | Confidence Thresholding | Detects variance spikes or low fidelity scores across segments. Flags incompatible media pairs and routes them to error logging to prevent corrupted output. |

---

## 📊 Handled Cases & Core Features

- **Audio Stream Deduplication & Ranking**: Evaluates stream properties (codecs, bitrate, sample rate, channel layout) based on configurable priority rules (e.g., FLAC > AAC > MP3) to retain only optimal tracks per language.
- **Subtitle Cleaning & Deduplication**: Prioritizes styled formats (e.g., ASS over raw SRT when text matches), cleans text streams, handles SDH/CC and dubtitle classification, and eliminates identical streams via MD5 hashing.
- **Track Metadata & Flag Assignment**: Automatically detects and tags default, forced, commentary, and audio-description streams, ensuring ISO 639 language code compliance.
- **Multi-Pass Remuxing Pipeline**: Executes initial stream extraction and normalization via FFmpeg, applies precise offset delays (`aresample`/`-itsoffset`), and performs final container assembly using `mkvmerge`.

---

## 🔮 Unhandled Cases & Agent Responsibilities

The system currently handles standard constant frame rate (CFR) releases with consistent cuts. Complex edge cases are actively being developed:

- **Unsegmented Cut Differences**: Extended vs. Theatrical cuts with missing or added scenes.
- **Variable Frame Rate (VFR) Drift**: Non-linear time drift across unaligned encodes.
- **Complex Container Incompatibilities**: Handling legacy codecs or damaged stream indices.

> **Agent Protocol**: You are expected to update this `AGENT.MD` file whenever you implement a new feature, extend detection stages, handle new edge cases, or adjust core architectural rules.

---

## 📜 Coding Directives & Constraints

### 1. Python Rules
- **Naming Conventions**: STRICTLY use `snake_case` for all new functions, variables, and methods. Do NOT use `camelCase`. Exclusive use of `snake_case` is required for all new variables, functions, and classes. Do not reproduce the `camelCase` that might still exist in the older legacy code.
- **Global State**: Route shared settings and runtime parameters exclusively through the `tools` module. Global state variables or configurations must be routed through the `tools` module.
- **Exception Handling**: Do NOT introduce blanket `try/except` blocks with `traceback.print_exc()` unless explicitly requested. Do not proactively generate global `try/except` blocks with `traceback.print_exc()`. Implementing global protections is at the exclusive discretion of the lead developer.
- **Type Hints**: Dynamic typing is preserved; PEP 484 type hints are optional.
- **Concurrency**: Continue using `threading` and `multiprocessing` for heavy workloads.

### 2. Rust Tooling (`audio_sync`)
- Code must pass `cargo check`, `cargo clippy`, and `cargo fmt` cleanly before submitting any changes.

### 3. Database & Configuration Constraints
- **Absolute Rule**: NEVER modify database structures or alter configuration keys (`config.json`, `config.ini`, `param_template.json`, or `tools/database.py`) without an explicit request in the prompt. It is strictly forbidden to modify the database structure or to add/remove configuration keys (in `config.json`, `config.ini`, `param_template.json`, or `tools/database.py`) without an explicit request in the prompt.

### 4. Restricted & Protected Areas
- **Production Scripts**: `init.sh` and `run.sh` are strictly READ-ONLY.
- **CI/CD & Environment**: Do NOT touch `.github/`, `.forgejo/`, or `Dockerfile`. Never read, modify, or alter the `.github/` and `.forgejo/` directories. The `Dockerfile` is read-only: it can be consulted to analyze the environment, but any modification is strictly forbidden.

## Project Architecture (Python & Rust)

- **Core logic**: The main entry point for merging operations is `mergeVideo.py`. Its role is to synchronize multiple video files based on language using audio correlation.
- **Orchestration**: `main.py` and `main_gestionar_show.py` act as wrappers and CLI entry points.
- **Role of Rust**: The external tool `fpcalc` performs an initial synchronization test. Then, the Rust code (which is an optimized rewrite of the second correlation) is called to perform the final adjustment without overloading the memory (preventing OOM errors).

## Testing and Execution

- **Python**: Do not generate or execute local tests with files outside `example/`. I will give you couple of files. They can be no compatible. If it is the case find why.
- **Rust**: Compilation via `cargo build` or `cargo check` is allowed and recommended to validate syntax.
- **Production scripts**: The `init.sh` and `run.sh` files are reserved for production. They can be read to understand the context, but must **never** be modified.
