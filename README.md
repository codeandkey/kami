# kami [![CI](https://github.com/codeandkey/kami/actions/workflows/rust.yml/badge.svg)](https://github.com/codeandkey/kami/actions/workflows/rust.yml) [![codecov](https://codecov.io/gh/codeandkey/kami/branch/master/graph/badge.svg?token=EmhIRCufkk)](https://codecov.io/gh/codeandkey/kami)
A portable chess engine powered by reinforcement learning.

## Installation

- Install Rust nightly using the instructions available at [rustup.rs](https://rustup.rs/).
- Install Python 3+ using the instructions available at [python.org](https://www.python.org/downloads/)

First install Torch and chess (for PGN export) for python:

```shell-session
python -m pip install torch chess
```

Build kami with CUDA support:

```bash
TORCH_CUDA_VERSION=cu111 cargo build --release
```

Launch kami and start training!

```bash
cargo run --release
```
