# VMSAM - AI Agent Directives

This file defines the strict rules that any AI assistant or code agent must follow when interacting with the VMSAM project.

## 1. Project Architecture (Python & Rust)

- **Core logic**: The main entry point for merging operations is `mergeVideo.py`. Its role is to synchronize multiple video files based on language using audio correlation.
- **Orchestration**: `main.py` and `main_gestionar_show.py` act as wrappers and CLI entry points.
- **Role of Rust**: The external tool `fpcalc` performs an initial synchronization test. Then, the Rust code (which is an optimized rewrite of the second correlation) is called to perform the final adjustment without overloading the memory (preventing OOM errors).

## 2. Coding Rules

### Python

- **Naming conventions**: Exclusive use of `snake_case` is required for all new variables, functions, and classes. Do not reproduce the `camelCase` that might still exist in the older legacy code.
- **Typing**: Maintain flexible dynamic typing (no strict obligation to add PEP 484 type hints).
- **Concurrency**: Continue using `threading` and `multiprocessing` for heavy workloads.
- **Exception handling**: Do not proactively generate global `try/except` blocks with `traceback.print_exc()`. Implementing global protections is at the exclusive discretion of the lead developer.
- **Global state**: Global state variables or configurations must be routed through the `tools` module.

### Rust

- Mandatory execution of `cargo clippy` to verify best practices.
- Mandatory execution of `cargo fmt` to format the code before making any proposal.

## 3. Database and Configuration

- **Absolute Rule**: It is strictly forbidden to modify the database structure or to add/remove configuration keys (in `config.json`, `config.ini`, `param_template.json`, or `tools/database.py`) without an explicit request in the prompt.

## 4. Testing and Execution

- **Python**: Do not generate or execute local tests. The video files are too heavy, and there is no mock environment available.
- **Rust**: Compilation via `cargo build` or `cargo check` is allowed and recommended to validate syntax.
- **Production scripts**: The `init.sh` and `run.sh` files are reserved for production. They can be read to understand the context, but must **never** be modified.

## 5. Restricted Areas (Strict Prohibitions)

- Never read, modify, or alter the `.github/` and `.forgejo/` directories.
- The `Dockerfile` is read-only: it can be consulted to analyze the environment, but any modification is strictly forbidden.
