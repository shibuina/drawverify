#!/usr/bin/env bash
# Create the `libero_sketch` conda env with LIBERO + this pipeline's deps.
# A dedicated env: robosuite 1.4 (LIBERO's pin) needs mujoco 2.3.x, not 3.x.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-libero_sketch}"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python=3.10
fi

mkdir -p "${HERE}/third_party"
if [ ! -d "${HERE}/third_party/LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${HERE}/third_party/LIBERO"
fi

# Real cmake binary via conda, first on PATH: egl_probe's build subprocess must not hit
# ~/.local/bin/cmake — that python shim breaks under pip build isolation (PYTHONNOUSERSITE
# hides its module). A pip-installed cmake is the same shim trap; conda's ELF binary isn't.
# Pinned <4: egl_probe's CMakeLists declares a <3.5 minimum that CMake 4.x refuses to build.
conda install -y -n "${ENV_NAME}" "cmake<4"
conda run -n "${ENV_NAME}" pip install wheel
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# CPU torch before robomimic pulls a multi-GB CUDA build (extraction never trains)
conda run -n "${ENV_NAME}" pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

conda run -n "${ENV_NAME}" pip install -r "${HERE}/third_party/LIBERO/requirements.txt"
# compat editable mode: LIBERO/libero has no __init__.py (implicit namespace package), so
# find_packages() yields nothing and a PEP 660 editable install produces an empty finder;
# compat mode puts the repo root on sys.path like legacy `setup.py develop` did.
conda run -n "${ENV_NAME}" pip install -e "${HERE}/third_party/LIBERO" \
    --config-settings editable_mode=compat

# first import prompts interactively for a dataset path; answer N -> default config
printf 'N\n' | conda run -n "${ENV_NAME}" --no-capture-output python -c "import libero.libero" || true

# robosuite 1.4 predates mujoco 3.x API changes; pip resolves 'mujoco' too new by default
conda run -n "${ENV_NAME}" pip install "mujoco==2.3.7"

conda run -n "${ENV_NAME}" pip install -r "${HERE}/requirements.txt"

# smoke check: LIBERO imports and benchmark registry resolves
conda run -n "${ENV_NAME}" python - <<'EOF'
from libero.libero import benchmark
d = benchmark.get_benchmark_dict()
assert {"libero_spatial", "libero_object", "libero_goal", "libero_10"} <= set(d)
print("LIBERO OK:", sorted(d.keys()))
EOF

echo
echo "Done. Next:"
echo "  conda activate ${ENV_NAME}"
echo "  export MUJOCO_GL=egl        # headless rendering (fallback: osmesa)"
echo "  python ${HERE}/scripts/00_download.py --suite all"
