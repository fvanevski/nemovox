# Merged-Main CPU Compilation, CLI & Regression Results (`7b4112cc31784953abbb041fd0b5d7997851841d`)

Executed in a clean detached worktree (`/workspace/nemovox-main-accept-lxOtyeHl`) at exact merged `main` (`7b4112cc31784953abbb041fd0b5d7997851841d`, tree `3d2cc429f9b49da3079df7d1fc178f6fd09d7987`) with `nvidia_transcribe` verified importing from `<worktree>/src/nvidia_transcribe/__init__.py`.

| Check | Python Environment | Result | Exit | Log |
| --- | --- | --- | ---: | --- |
| `python -m compileall -q src/nvidia_transcribe` | `.venv` (Python 3.13.15) | Clean | `0` | [`logs/compileall.txt`](logs/compileall.txt) |
| `pytest -v tests/test_qwen_session_and_policy.py tests/test_qwen_fallback.py tests/test_studio_api.py` | `.venv` (Python 3.13.15) | **`57 passed, 1 warning in 7.88s`** | `0` | [`logs/focused_pytest.txt`](logs/focused_pytest.txt) |
| `pytest -q -m "not gpu"` (full CPU suite) | `.venv` (Python 3.13.15) | **`120 passed, 1 warning in 8.75s`** | `0` | [`logs/pytest_venv_cpu.txt`](logs/pytest_venv_cpu.txt) |
| `pytest -q -m "not gpu"` (full CPU suite) | `.venv-studio` (Python 3.13) | **`120 passed in 8.74s`** | `0` | [`logs/pytest_venv_studio_cpu.txt`](logs/pytest_venv_studio_cpu.txt) |
| CLI `--help` (`process-media`, `generate-subtitles`, `recover-qwen-asr`) | `.venv` (Python 3.13.15) | Verified defaults (`alignment_fallback=off`, `qwen_policy=subtitle`, `qwen_device=cuda:0`, `qwen_gpu_mode=available`, explicit `managed` & `recover-qwen-asr`) | `0` | [`logs/cli_help.txt`](logs/cli_help.txt) |
