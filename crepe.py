"""
DDSP import compatibility stub for local timbre transfer.

The local DDSP flute pipeline uses librosa.pyin for F0 extraction instead of
CREPE, but ddsp.spectral_ops imports `crepe` unconditionally. This stub keeps
that import working without pulling in the legacy CREPE dependency stack.
"""


class _Core:
    models = {"full": None, "large": None, "medium": None, "small": None, "tiny": None}

    @staticmethod
    def build_and_load_model(*args, **kwargs):
        raise RuntimeError(
            "CREPE is not installed in this environment. "
            "The local DDSP flute pipeline uses librosa.pyin instead."
        )


core = _Core()


def predict(*args, **kwargs):
    raise RuntimeError(
        "CREPE-based f0 extraction is unavailable in this environment. "
        "Use the bundled DDSP flute transfer script, which relies on librosa.pyin."
    )
