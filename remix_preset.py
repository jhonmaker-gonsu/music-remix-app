"""
LoveLetter リミックスプリセット — 2026-04-01

入力: /Users/gon/Desktop/LoveLetter.mp3
"""

# 最初の設定 (v1)
PRESET_V1 = {
    "quantize_grid": "64th",
    "quantize_strength": 1.0,
    "formant": {
        "vocals": +2.0,
    },
    "stems": {
        "drums": {
            "volume": 2.2,
            "distortion_drive_db": 5,
            "delay_seconds": 0.1,
            "delay_feedback": 0.1,
            "delay_mix": 0.05,
            "comp_threshold_db": -18,
            "comp_ratio": 6,
            "comp_attack_ms": 2,
            "comp_release_ms": 40,
            "gain_db": 6,
        },
        "bass": {
            "volume": 1.0,
            "lowpass_hz": 800,
            "distortion_drive_db": 5,
            "delay_seconds": 0.1,
            "delay_feedback": 0.1,
            "delay_mix": 0.10,
            "comp_threshold_db": -20,
            "comp_ratio": 6,
            "comp_attack_ms": 10,
            "comp_release_ms": 60,
            "gain_db": 4,
        },
        "vocals": {
            "volume": 0.9,
            "distortion_drive_db": 15,
            "delay_seconds": 0.1,
            "delay_feedback": 0.1,
            "delay_mix": 0.05,
            "comp_threshold_db": -18,
            "comp_ratio": 5,
            "comp_attack_ms": 8,
            "comp_release_ms": 80,
            "gain_db": 5,
        },
        "other": {
            "volume": 0.7,
            "distortion_drive_db": 2,
            "delay_seconds": 0.1,
            "delay_feedback": 0.1,
            "delay_mix": 0.10,
            "comp_threshold_db": -20,
            "comp_ratio": 5,
            "comp_attack_ms": 10,
            "comp_release_ms": 60,
            "gain_db": 4,
        },
    },
    "master": {
        "reverb_room_size": 0.3,
        "reverb_wet": 0.15,
        "reverb_dry": 0.85,
        "reverb_width": 1.0,
        "limiter_threshold_db": -0.1,
        "limiter_release_ms": 80,
    },
}

# 現在の設定 (v2) — harvest F0, フォルマント-4, ボーカルにリバーブ追加
PRESET_V2 = {
    "quantize_grid": None,  # クオンタイズなし
    "quantize_strength": 1.0,

    "formant": {
        "vocals": -4.0,  # 半音 (harvest F0推定使用)
    },
    "formant_f0_method": "harvest",  # dio より裏声に強い

    "stems": {
        "drums": {
            "volume": 2.2,
            "distortion_drive_db": 5,
            "delay_seconds": 0.1,
            "delay_feedback": 0.1,
            "delay_mix": 0.05,
            "comp_threshold_db": -18,
            "comp_ratio": 6,
            "comp_attack_ms": 2,
            "comp_release_ms": 40,
            "gain_db": 3,
        },
        "bass": {
            "volume": 1.0,
            "distortion_drive_db": 5,
            "lowpass_hz": 800,
            "delay_seconds": 0.1,
            "delay_feedback": 0.1,
            "delay_mix": 0.10,
            "comp_threshold_db": -20,
            "comp_ratio": 6,
            "comp_attack_ms": 10,
            "comp_release_ms": 60,
            "gain_db": 2,
        },
        "vocals": {
            "volume": 1.2,
            "distortion_drive_db": 2,
            "delay_seconds": 0.12,
            "delay_feedback": 0.08,
            "delay_mix": 0.06,              # かなり薄めディレイ
            "reverb_room_size": 0.55,
            "reverb_wet": 0.30,             # 少し濃いめリバーブ
            "reverb_dry": 0.70,
            "comp_threshold_db": -18,
            "comp_ratio": 5,
            "comp_attack_ms": 8,
            "comp_release_ms": 80,
            "gain_db": 5,
        },
        "other": {
            "volume": 0.7,
            "distortion_drive_db": 2,
            "delay_seconds": 0.1,
            "delay_feedback": 0.1,
            "delay_mix": 0.10,
            "comp_threshold_db": -20,
            "comp_ratio": 5,
            "comp_attack_ms": 10,
            "comp_release_ms": 60,
            "gain_db": 2,
        },
    },

    "master": {
        "reverb_room_size": 0.3,
        "reverb_wet": 0.15,
        "reverb_dry": 0.85,
        "reverb_width": 1.0,
        "limiter_threshold_db": -0.1,
        "limiter_release_ms": 80,
    },
}

# 現在使用中のプリセット
PRESET = PRESET_V2
