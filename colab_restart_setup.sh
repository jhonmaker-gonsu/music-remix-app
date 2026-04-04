#!/usr/bin/env bash
set -euo pipefail

# Rebuild the DDSP runtime on Google Colab after a runtime restart.
# Usage in Colab:
#   %cd /content/music-remix-app
#   !bash /content/music-remix-app/colab_restart_setup.sh

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MAMBA_BIN="${MAMBA_BIN:-/usr/local/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/content/.micromamba}"
ENV_NAME="${ENV_NAME:-ddsp310}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-ddsp-colab}"
MODELS_DIR="${MODELS_DIR:-${PROJECT_DIR}/ddsp_models}"

if [[ ! -f "${PROJECT_DIR}/ddsp_setup.py" ]]; then
  echo "ddsp_setup.py not found under PROJECT_DIR=${PROJECT_DIR}" >&2
  echo "Run this from the repository root, or pass PROJECT_DIR=/content/music-remix-app." >&2
  exit 1
fi

echo "[1/5] Installing OS packages..."
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsndfile1 libsndfile1-dev

if [[ ! -x "${MAMBA_BIN}" ]]; then
  echo "[2/5] Installing micromamba..."
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' EXIT
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "${tmpdir}" bin/micromamba >/dev/null
  install -m 755 "${tmpdir}/bin/micromamba" "${MAMBA_BIN}"
else
  echo "[2/5] micromamba already installed."
fi

echo "[3/5] Creating Python 3.10 env..."
export MAMBA_ROOT_PREFIX
"${MAMBA_BIN}" create -y -n "${ENV_NAME}" python=3.10 >/dev/null

echo "[4/5] Running ddsp_setup.py..."
"${MAMBA_BIN}" run -n "${ENV_NAME}" python "${PROJECT_DIR}/ddsp_setup.py" \
  --venv "${VENV_DIR}" \
  --models-dir "${MODELS_DIR}" \
  --model Flute

echo "[5/5] Smoke test..."
"${VENV_DIR}/bin/python" - <<'PY'
import ddsp
import gin
import librosa
import soundfile
import tensorflow as tf

print("python ok")
print("tensorflow", tf.__version__)
print("ddsp", ddsp.__version__ if hasattr(ddsp, "__version__") else "imported")
print("gin", gin.__version__ if hasattr(gin, "__version__") else "imported")
print("librosa", librosa.__version__)
print("soundfile", soundfile.__version__)
PY

cat <<EOF

DDSP Colab setup finished.

Next step examples:
  ${VENV_DIR}/bin/python "${PROJECT_DIR}/ddsp_flute_transfer.py" --help
  ${VENV_DIR}/bin/python "${PROJECT_DIR}/ddsp_flute_transfer.py" \\
    --input /content/input.wav \\
    --output /content/output_flute.wav \\
    --model-dir "${MODELS_DIR}/solo_flute_ckpt" \\
    --backend legacy \\
    --vst-model Flute

EOF
