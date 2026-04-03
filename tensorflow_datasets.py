"""
Minimal tensorflow_datasets stub for local DDSP inference.

DDSP's training package imports tensorflow_datasets at module import time, but
the local flute-transfer pipeline never calls into TFDS-backed data loading.
This stub keeps those imports lightweight and avoids native dependency issues
on Apple Silicon.
"""


def load(*args, **kwargs):
    raise RuntimeError(
        "tensorflow_datasets の実データ読み込みはこのローカル推論では未対応です。"
    )
