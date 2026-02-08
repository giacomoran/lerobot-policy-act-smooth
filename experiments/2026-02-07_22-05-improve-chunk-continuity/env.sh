#!/bin/bash
# Shared environment setup for experiments.
# Source this before running any experiment script.
#
# Workaround: nix develop --command doesn't set LD_LIBRARY_PATH correctly,
# so we manually add libstdc++, zlib, and NVIDIA driver libs.

DIR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Create nvidia-libs symlink dir if needed
if [ ! -d /tmp/nvidia-libs ]; then
    mkdir -p /tmp/nvidia-libs
    for lib in /usr/lib/x86_64-linux-gnu/libcuda.so* /usr/lib/x86_64-linux-gnu/libnvidia*.so*; do
        [ -e "$lib" ] && ln -sf "$lib" "/tmp/nvidia-libs/$(basename "$lib")"
    done
fi

# Set LD_LIBRARY_PATH for libstdc++, zlib, and NVIDIA driver
export LD_LIBRARY_PATH="/nix/store/xm08aqdd7pxcdhm0ak6aqb1v7hw5q6ri-gcc-14.3.0-lib/lib:/nix/store/f2q5ld1nipl8w1r2w8m6azhlm2varqgb-zlib-1.3.1/lib:/tmp/nvidia-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Activate venv
source "${DIR_ROOT}/.venv/bin/activate"

echo "Environment ready: $(python -c 'import torch; print(f"CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}")')"
