# kami [![CI](https://github.com/codeandkey/kami/actions/workflows/rust.yml/badge.svg)](https://github.com/codeandkey/kami/actions/workflows/rust.yml) [![Coverage Status](https://coveralls.io/repos/github/codeandkey/kami/badge.svg?branch=master)](https://coveralls.io/github/codeandkey/kami?branch=master)
A portable chess engine powered by reinforcement learning.

## Installation

- Install Rust nightly using the instructions available at [rustup.rs](https://rustup.rs/).
- Install Python 3+ using the instructions available at [python.org](https://www.python.org/downloads/)

Install Torch and chess (for PGN export) for python:

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
