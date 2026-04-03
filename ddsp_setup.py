#!/usr/bin/env python3
"""
ddsp_setup.py — DDSP flute transfer 用の専用環境とモデルを準備する
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path


MODEL_PREFIXES = {
    "Flute": "models/timbre_transfer_colab/2021-07-08/solo_flute_ckpt/",
}

CORE_PACKAGES = [
    "cloudml-hypertune<=0.1.0.dev6",
    "tensorflow-macos==2.11.0",
    "tensorflow-probability==0.19.0",
    "numpy<1.24",
    "scipy<=1.10.1",
    "librosa<=0.10",
    "matplotlib<3.10",
    "gin-config>=0.3.0",
    "soundfile",
    "pydub<=0.25.1",
    "mir_eval<=0.7",
    "note_seq<0.0.4",
    "dill<=0.3.4",
    "future",
    "absl-py",
    "protobuf<=3.20",
    "google-cloud-storage",
]

DDSP_PACKAGE = "ddsp==3.7.0"


def choose_python() -> str:
    for candidate in ("python3.9", "python3.10", "python3"):
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError("python3.9 / python3.10 / python3 が見つかりません。")


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def ensure_venv(venv_dir: Path) -> Path:
    python_bin = venv_dir / "bin" / "python"
    if python_bin.exists():
        return python_bin

    base_python = choose_python()
    run([base_python, "-m", "venv", str(venv_dir)])
    return python_bin


def ensure_packages(venv_python: Path, project_dir: Path) -> None:
    marker = venv_python.parent.parent / ".ddsp_ready.json"
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    def verify_runtime() -> bool:
        verify = (
            "import sys; "
            f"sys.path.insert(0, {project_dir.as_posix()!r}); "
            "import tensorflow as tf; import ddsp; import ddsp.training; import gin; import librosa; import soundfile; "
            "print(tf.__version__)"
        )
        try:
            run([str(venv_python), "-c", verify], env=env)
            return True
        except Exception:
            return False

    if marker.exists() and verify_runtime():
        return

    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools<81", "wheel"], env=env)
    run([str(venv_python), "-m", "pip", "install", *CORE_PACKAGES], env=env)
    run([str(venv_python), "-m", "pip", "install", DDSP_PACKAGE, "--no-deps"], env=env)

    # Apple Silicon なら Metal を試す。失敗しても継続。
    try:
        run([str(venv_python), "-m", "pip", "install", "tensorflow-metal==0.7.0"], env=env)
    except Exception:
        print("tensorflow-metal はスキップします。")

    if not verify_runtime():
        raise RuntimeError("DDSP実行環境の検証に失敗しました。")
    marker.write_text(json.dumps({"python": str(venv_python), "package": DDSP_PACKAGE}, indent=2), encoding="utf-8")


def list_bucket_objects(prefix: str) -> list[dict]:
    url = "https://storage.googleapis.com/storage/v1/b/ddsp/o?prefix=" + urllib.parse.quote(prefix, safe="")
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("items", [])


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def ensure_model(model_name: str, models_dir: Path) -> Path:
    prefix = MODEL_PREFIXES[model_name]
    model_dir = models_dir / f"solo_{model_name.lower()}_ckpt"
    expected = [
        model_dir / "operative_config-0.gin",
        model_dir / "dataset_statistics.pkl",
    ]
    if all(path.exists() for path in expected):
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    for item in list_bucket_objects(prefix):
        name = item["name"]
        relative = name.removeprefix(prefix)
        if not relative:
            continue
        media_url = item.get("mediaLink")
        if not media_url:
            continue
        download_file(media_url, model_dir / relative)
    return model_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="DDSP flute transfer の準備")
    parser.add_argument("--venv", required=True, help="専用venvディレクトリ")
    parser.add_argument("--models-dir", required=True, help="DDSPモデル保存先")
    parser.add_argument("--model", default="Flute", choices=sorted(MODEL_PREFIXES.keys()))
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    venv_dir = Path(args.venv).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()

    venv_python = ensure_venv(venv_dir)
    ensure_packages(venv_python, project_dir)
    model_dir = ensure_model(args.model, models_dir)
    print(f"DDSP ready: python={venv_python}")
    print(f"Model ready: {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
