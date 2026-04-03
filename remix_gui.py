#!/usr/bin/env python3
"""
remix_gui.py — ステム別エフェクト調整GUI
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from pedalboard import (
    Bitcrush,
    Clipping,
    Compressor,
    Delay,
    Distortion,
    Gain,
    LadderFilter,
    Limiter,
    LowpassFilter,
    Pedalboard,
    Reverb,
)

try:
    from PySide6.QtCore import QObject, Qt, QTimer, Signal
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QInputDialog,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QProgressBar,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "PySide6 が必要です。`python3 -m pip install PySide6` を実行してください。"
    ) from exc

sys.path.insert(0, str(Path(__file__).parent))
try:
    import importlib
    import music_remix

    importlib.reload(music_remix)
    from music_remix import instrumentize_vocal, shift_formant

    HAS_FORMANT = True
    HAS_INSTRUMENTIZE = True
except Exception:
    HAS_FORMANT = False
    HAS_INSTRUMENTIZE = False


class TaskSignals(QObject):
    status = Signal(str)
    progress = Signal(int)  # 0-100
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class SliderControl(QWidget):
    def __init__(self, label: str, minimum: float, maximum: float, default: float, step: float):
        super().__init__()
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.step = float(step)
        self._max_index = int(round((self.maximum - self.minimum) / self.step))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.label = QLabel(label)
        self.label.setMinimumWidth(110)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._max_index)
        self.slider.setPageStep(max(1, self._max_index // 10))
        self.slider.setSingleStep(1)
        self.slider.setTracking(True)
        self.slider.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.slider, 1)

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(self.minimum, self.maximum)
        self.spinbox.setSingleStep(self.step)
        self.spinbox.setDecimals(self._decimals_for_step(self.step))
        self.spinbox.setAlignment(Qt.AlignRight)
        self.spinbox.setFixedWidth(92)
        layout.addWidget(self.spinbox)

        self.slider.valueChanged.connect(self._sync_from_slider)
        self.spinbox.valueChanged.connect(self._sync_from_spinbox)
        self.set_value(default)

    @staticmethod
    def _decimals_for_step(step: float) -> int:
        text = f"{step:.10f}".rstrip("0")
        if "." not in text:
            return 0
        return len(text.split(".", 1)[1])

    def _index_to_value(self, index: int) -> float:
        value = self.minimum + (index * self.step)
        decimals = self._decimals_for_step(self.step)
        return round(value, decimals)

    def _value_to_index(self, value: float) -> int:
        return int(round((value - self.minimum) / self.step))

    def _sync_from_slider(self, index: int) -> None:
        value = self._index_to_value(index)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)

    def _sync_from_spinbox(self, value: float) -> None:
        index = max(0, min(self._max_index, self._value_to_index(value)))
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)

    def value(self) -> float:
        return self.spinbox.value()

    def set_value(self, value: float) -> None:
        index = max(0, min(self._max_index, self._value_to_index(value)))
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        self.slider.setValue(index)
        self.spinbox.setValue(self._index_to_value(index))
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)


class StemControl(QGroupBox):
    def __init__(self, name: str):
        super().__init__(f" {name.upper()} ")
        self.name = name
        self.controls: dict[str, SliderControl] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        for label, key, minimum, maximum, default, step in self._defaults():
            control = SliderControl(label, minimum, maximum, default, step)
            layout.addWidget(control)
            self.controls[key] = control
        layout.addStretch(1)

    def _defaults(self):
        base = [
            ("Volume", "volume", 0.0, 4.0, 1.0, 0.1),
            ("Distortion", "dist_db", 0.0, 30.0, 0.0, 1.0),
            ("Bitcrush bit", "bitcrush", 0, 32, 0, 1),        # ローファイ: bit数を下げてデジタル劣化
            ("Clipping dB", "clipping_db", -30, 0, 0, 1),     # ハードクリップ: 指定dBで波形を切る
            ("Ladder Hz", "ladder_hz", 0, 20000, 0, 100),      # アナログシンセ風フィルター: 0=OFF
            ("Lowpass Hz", "lowpass_hz", 200, 20000, 20000, 100),
            ("Delay ms", "delay_ms", 0, 500, 100, 10),
            ("Delay FB %", "delay_fb", 0, 80, 10, 1),
            ("Delay Mix %", "delay_mix", 0, 80, 0, 1),
            ("Reverb Room", "reverb_room", 0.0, 1.0, 0.0, 0.05),
            ("Reverb Wet %", "reverb_wet", 0, 100, 0, 1),
            ("Comp Thresh", "comp_thresh", -60, 0, -18, 1),
            ("Comp Ratio", "comp_ratio", 1.0, 30.0, 4.0, 0.5),
            ("Comp Atk ms", "comp_attack", 0.1, 100, 5, 0.5),
            ("Comp Rel ms", "comp_release", 10, 500, 50, 5),
            ("Gain dB", "gain_db", -12, 18, 0, 0.5),
        ]
        if self.name == "vocals":
            base.insert(1, ("Formant 半音", "formant", -12.0, 12.0, 0.0, 0.5))
            base.insert(2, ("楽器化 %", "instrumentize", 0, 100, 0, 1))
            base.insert(3, ("子音抑制 %", "breath_reduce", 0, 100, 75, 1))
            base.insert(4, ("暗さ %", "tone_darken", 0, 100, 35, 1))
            base.insert(5, ("子音ゲート %", "consonant_suppress", 0, 100, 65, 1))
            base.insert(6, ("高域ぼかし %", "modulation_blur", 0, 100, 45, 1))
            base.insert(7, ("激歪み %", "grit_drive", 0, 100, 0, 1))
            base.insert(8, ("ロボ変調 %", "robot_mod", 0, 100, 0, 1))
        return base

    def get_params(self) -> dict[str, float]:
        return {key: control.value() for key, control in self.controls.items()}

    def set_params(self, params: dict[str, float]) -> None:
        for key, value in params.items():
            if key in self.controls:
                self.controls[key].set_value(value)


class RemixGUI(QMainWindow):
    STEM_NAMES = ["drums", "bass", "vocals", "other"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Remix GUI")
        self.resize(1180, 920)
        self.setMinimumSize(900, 680)

        self.stems_raw: dict[str, np.ndarray] = {}
        self.sr = 44100
        self.playing = False
        self._cancel = False  # 処理中断フラグ
        self.track_name = ""  # 読み込んだ曲名
        self.export_format = "wav"
        self.action_buttons: list[QPushButton] = []
        self._active_tasks: list[tuple[threading.Thread, TaskSignals]] = []
        self.presets_dir = Path(__file__).parent / "presets"
        self.presets_dir.mkdir(exist_ok=True)
        self.ddsp_venv_dir = Path(__file__).parent / ".venv-ddsp"
        self.ddsp_models_dir = Path(__file__).parent / "ddsp_models"

        self._build_ui()
        self._refresh_preset_combo()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        self.load_stems_button = self._make_button("ステム読み込み", self._load_stems, min_width=150)
        self.load_audio_button = self._make_button("MP3/WAV 読み込み", self._load_audio, min_width=170)
        top_row.addWidget(self.load_stems_button)
        top_row.addWidget(self.load_audio_button)

        self.status_label = QLabel("準備完了 - ステムまたは音声ファイルを読み込んでください")
        self.status_label.setWordWrap(True)
        self.status_label.setFrameShape(QFrame.StyledPanel)
        self.status_label.setMinimumHeight(68)
        self.status_label.setMargin(10)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_row.addWidget(self.status_label, 1)
        root.addLayout(top_row)

        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.hide()
        root.addWidget(self.progress_bar)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.preview_button = self._make_button("プレビュー (30秒)", self._preview, min_width=170, bold=True)
        self.stop_button = self._make_button("停止", self._stop_preview, min_width=90)
        self.stop_button.setEnabled(False)
        self.export_button = self._make_button("書き出し", self._export, min_width=130, bold=True)
        self.ddsp_button = self._make_button("DDSP VST 楽器化", self._run_ddsp_flute, min_width=170, bold=True)
        self.preset_button = self._make_button("V2 プリセット", self._load_v2_preset, min_width=130)
        self.instrument_preset_button = self._make_button(
            "歌メロ楽器化", self._load_instrument_preset, min_width=130
        )
        self.reset_button = self._make_button("リセット", self._reset_all, min_width=110)

        action_row.addWidget(self.preview_button)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self._build_format_box())
        action_row.addWidget(self.export_button)
        action_row.addWidget(self.ddsp_button)
        action_row.addStretch(1)

        # プリセットプルダウン + 保存/削除
        action_row.addWidget(QLabel("プリセット:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(180)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        action_row.addWidget(self.preset_combo)
        self.save_preset_button = self._make_button("保存", self._save_preset, min_width=70)
        self.delete_preset_button = self._make_button("削除", self._delete_preset, min_width=70)
        action_row.addWidget(self.save_preset_button)
        action_row.addWidget(self.delete_preset_button)
        action_row.addWidget(self.preset_button)
        action_row.addWidget(self.instrument_preset_button)
        action_row.addWidget(self.reset_button)
        root.addLayout(action_row)

        # ── 3段目: プレビュー開始位置 + ミュート + ボーカルなし書き出し ──
        row3 = QHBoxLayout()
        row3.setSpacing(10)

        # プレビュー開始位置
        row3.addWidget(QLabel("プレビュー開始:"))
        self.preview_offset = QSpinBox()
        self.preview_offset.setRange(0, 600)
        self.preview_offset.setValue(0)
        self.preview_offset.setSuffix(" 秒")
        self.preview_offset.setMinimumWidth(100)
        row3.addWidget(self.preview_offset)

        row3.addSpacing(20)

        # ステムミュートチェックボックス
        row3.addWidget(QLabel("ミュート:"))
        self.mute_checks: dict[str, QCheckBox] = {}
        for name in self.STEM_NAMES:
            cb = QCheckBox(name.upper())
            cb.setChecked(False)
            self.mute_checks[name] = cb
            row3.addWidget(cb)

        row3.addSpacing(20)

        # ボーカルなし書き出し
        self.export_no_vocal_button = self._make_button(
            "ボーカルなし書き出し", self._export_no_vocals, min_width=170)
        row3.addWidget(self.export_no_vocal_button)

        row3.addStretch(1)
        root.addLayout(row3)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        self.stem_controls: dict[str, StemControl] = {}
        for name in self.STEM_NAMES:
            control = StemControl(name)
            self.stem_controls[name] = control
            self.tabs.addTab(self._wrap_in_scroll(control), name.upper())

        self.master_sliders = {
            "m_reverb_room": SliderControl("Reverb Room", 0.0, 1.0, 0.4, 0.05),
            "m_reverb_wet": SliderControl("Reverb Wet %", 0, 100, 25, 1),
            "m_limiter_db": SliderControl("Limiter dBFS", -6.0, 0.0, -0.1, 0.1),
        }
        master_box = QGroupBox(" MASTER ")
        master_layout = QVBoxLayout(master_box)
        master_layout.setContentsMargins(12, 12, 12, 12)
        master_layout.setSpacing(8)
        for slider in self.master_sliders.values():
            master_layout.addWidget(slider)
        master_layout.addStretch(1)
        self.tabs.addTab(self._wrap_in_scroll(master_box), "MASTER")
        root.addWidget(self.tabs, 1)

        self.setCentralWidget(central)

    def _make_button(self, text: str, handler, min_width: int = 120, bold: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(handler)
        button.setMinimumHeight(44)
        button.setMinimumWidth(min_width)
        button.setAutoDefault(False)
        button.setDefault(False)
        button.setFocusPolicy(Qt.NoFocus)
        if bold:
            font = button.font()
            font.setBold(True)
            button.setFont(font)
        self.action_buttons.append(button)
        return button

    def _build_format_box(self) -> QGroupBox:
        box = QGroupBox("出力")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.format_group = QButtonGroup(self)
        wav = QRadioButton("WAV")
        mp3 = QRadioButton("MP3")
        wav.setChecked(True)
        wav.toggled.connect(lambda checked: checked and self._set_export_format("wav"))
        mp3.toggled.connect(lambda checked: checked and self._set_export_format("mp3"))
        self.format_group.addButton(wav)
        self.format_group.addButton(mp3)
        layout.addWidget(wav)
        layout.addWidget(mp3)
        return box

    def _set_export_format(self, value: str) -> None:
        self.export_format = value

    def _wrap_in_scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(widget)
        return scroll

    @staticmethod
    def _normalize_output_path(outpath: str, fmt: str) -> str:
        path = Path(outpath)
        if path.suffix.lower() == f".{fmt}":
            return str(path)
        if path.suffix:
            return str(path.with_suffix(f".{fmt}"))
        return str(path.with_name(f"{path.name}.{fmt}"))

    def _run_task(self, task_fn, on_result=None, on_finished=None) -> None:
        signals = TaskSignals()
        signals.status.connect(self._set_status)
        signals.progress.connect(self._update_progress)
        signals.error.connect(lambda text: self._set_status(f"エラー: {text}"))
        if on_result is not None:
            signals.result.connect(on_result)

        thread: threading.Thread | None = None

        def cleanup() -> None:
            nonlocal thread
            if thread is not None:
                self._active_tasks = [item for item in self._active_tasks if item[0] is not thread]
            if on_finished is not None:
                on_finished()

        signals.finished.connect(cleanup)

        def runner() -> None:
            try:
                result = task_fn(signals)
                if result is not None:
                    signals.result.emit(result)
            except Exception as exc:
                signals.error.emit(str(exc))
            finally:
                signals.finished.emit()

        thread = threading.Thread(target=runner, daemon=True)
        self._active_tasks.append((thread, signals))
        thread.start()

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _update_progress(self, value: int) -> None:
        if value <= 0:
            self.progress_bar.hide()
            self.progress_bar.setValue(0)
        elif value >= 100:
            self.progress_bar.setValue(100)
            QTimer.singleShot(500, self.progress_bar.hide)
        else:
            self.progress_bar.show()
            self.progress_bar.setValue(value)

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        for button in self.action_buttons:
            if button is self.stop_button:
                continue
            button.setEnabled(enabled)
        self.stop_button.setEnabled(self.playing)

    def _restore_focus(self) -> None:
        for delay in (0, 120):
            QTimer.singleShot(delay, self._activate_window)

    def _activate_window(self) -> None:
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.ActiveWindowFocusReason)

    def _load_stems(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "ステムフォルダを選択")
        self._restore_focus()
        if not folder:
            return
        folder_path = Path(folder)
        found = {}
        for name in self.STEM_NAMES:
            for pattern in (f"step1_stem_{name}.wav", f"{name}.wav"):
                candidate = folder_path / pattern
                if candidate.exists():
                    audio, sr = sf.read(candidate, dtype="float64")
                    found[name] = audio
                    self.sr = sr
                    break
        if len(found) < 4:
            QMessageBox.critical(self, "エラー", f"4ステム中{len(found)}個のみ\n{folder_path}")
            return
        self.stems_raw = found
        self.track_name = folder_path.name
        self._set_status(f"読み込み完了: {folder_path.name} ({self.sr}Hz)")

    def _load_audio(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "音声ファイルを選択",
            "",
            "Audio (*.mp3 *.wav *.flac *.m4a *.ogg);;All Files (*)",
        )
        self._restore_focus()
        if not filepath:
            return

        filename = Path(filepath).name
        self._set_status(f"demucs 分離中... {filename} (数分かかります)")
        self._set_action_buttons_enabled(False)

        def task(signals: TaskSignals):
            tmp_dir = tempfile.mkdtemp(prefix="remix_gui_")
            try:
                # demucsをsubprocessで実行し、stderrから進捗をパース
                import re
                proc = subprocess.Popen(
                    [sys.executable, "-m", "demucs", "-o", tmp_dir, filepath],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                signals.progress.emit(1)
                for line in proc.stdout:
                    # demucsのtqdmプログレスから %を抽出
                    m = re.search(r'(\d+)%\|', line)
                    if m:
                        pct = int(m.group(1))
                        signals.progress.emit(max(1, min(95, pct)))
                        signals.status.emit(f"demucs 分離中... {pct}%")
                proc.wait()
                signals.progress.emit(95)
                if proc.returncode != 0:
                    raise RuntimeError(f"demucs failed (exit code {proc.returncode})")
                track_name = Path(filepath).stem
                stem_dir = Path(tmp_dir) / "htdemucs" / track_name
                found = {}
                sr = self.sr
                for stem_file in sorted(stem_dir.glob("*.wav")):
                    if stem_file.stem not in self.STEM_NAMES:
                        continue
                    audio, sr = sf.read(stem_file, dtype="float64")
                    found[stem_file.stem] = audio
                if len(found) < 4:
                    raise RuntimeError("ステムが4つ見つかりません")
                return found, sr, track_name, filepath
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        def on_result(result) -> None:
            found, sr, track_name, original_path = result
            self.stems_raw = found
            self.sr = sr
            self.track_name = track_name
            # 曲名フォルダを作成してステムと元ファイルを保存
            stem_out = Path(__file__).parent / f"remix_{track_name}"
            stem_out.mkdir(exist_ok=True)
            for sname, saudio in found.items():
                sf.write(str(stem_out / f"{sname}.wav"), saudio, sr, subtype="PCM_24")
            # 元ファイルをフォルダに移動（コピーではなく移動）
            src = Path(original_path)
            dst = stem_out / src.name
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
            self._update_progress(100)
            self._set_status(f"分離完了: {track_name} ({self.sr}Hz) — 保存先: remix_{track_name}/")

        self._run_task(task, on_result=on_result, on_finished=lambda: self._set_action_buttons_enabled(True))

    def _collect_params(self):
        stem_params = {name: self.stem_controls[name].get_params() for name in self.STEM_NAMES}
        master_params = {key: control.value() for key, control in self.master_sliders.items()}
        return stem_params, master_params

    def _process_stems(self, preview_seconds=None, preview_offset=0,
                       stem_params=None, master_params=None,
                       status_callback=None, mute_stems=None):
        if not self.stems_raw:
            return None
        missing = [name for name in self.STEM_NAMES if name not in self.stems_raw]
        if missing:
            if status_callback is not None:
                status_callback(f"エラー: ステムが不足: {', '.join(missing)}")
            return None
        if mute_stems is None:
            mute_stems = set()

        sr = self.sr
        processed = {}
        for name in self.STEM_NAMES:
            if self._cancel:
                if status_callback is not None:
                    status_callback("中断されました")
                return None
            if name in mute_stems:
                # ミュートされたステムはゼロ信号
                ref = self.stems_raw[name]
                if preview_seconds:
                    n_samples = int(sr * preview_seconds)
                    offset_samples = int(sr * preview_offset)
                    processed[name] = np.zeros_like(ref[offset_samples:offset_samples + n_samples])
                else:
                    processed[name] = np.zeros_like(ref)
                continue
            params = stem_params[name]
            if preview_seconds:
                n_samples = int(sr * preview_seconds)
                offset_samples = int(sr * preview_offset)
                audio = self.stems_raw[name][offset_samples:offset_samples + n_samples].copy()
            else:
                audio = self.stems_raw[name].copy()

            if name == "vocals" and HAS_FORMANT and abs(params.get("formant", 0)) > 0.1:
                if status_callback is not None:
                    status_callback(f"フォルマント処理中... ({name})")
                audio = shift_formant(audio, sr, params["formant"])

            if name == "vocals" and HAS_INSTRUMENTIZE and params.get("instrumentize", 0) > 0.1:
                if status_callback is not None:
                    status_callback(f"歌メロ楽器化処理中... ({name})")
                audio = instrumentize_vocal(
                    audio,
                    sr,
                    amount=params["instrumentize"] / 100.0,
                    breath_reduction=params.get("breath_reduce", 75) / 100.0,
                    tone_darken=params.get("tone_darken", 35) / 100.0,
                    consonant_suppress=params.get("consonant_suppress", 65) / 100.0,
                    modulation_blur=params.get("modulation_blur", 45) / 100.0,
                    grit_drive=params.get("grit_drive", 0) / 100.0,
                    robot_mod=params.get("robot_mod", 0) / 100.0,
                )

            effects = []
            if params["dist_db"] > 0.1:
                effects.append(Distortion(drive_db=params["dist_db"]))
            # Bitcrush: bit数を下げてローファイ化 (32=OFF, 低いほど劣化)
            if 0 < params.get("bitcrush", 0) < 32:
                effects.append(Bitcrush(bit_depth=params["bitcrush"]))
            # Clipping: 指定dBで波形をハードクリップ (0=OFF, 負の値で強い歪み)
            if params.get("clipping_db", 0) < -0.1:
                effects.append(Clipping(threshold_db=params["clipping_db"]))
            # LadderFilter: アナログシンセ風レゾナンスフィルター (0=OFF)
            if params.get("ladder_hz", 0) > 100:
                effects.append(LadderFilter(
                    mode=LadderFilter.Mode.LPF24,
                    cutoff_hz=params["ladder_hz"],
                    resonance=0.5,
                ))
            if params["lowpass_hz"] < 19000:
                effects.append(LowpassFilter(cutoff_frequency_hz=params["lowpass_hz"]))
            if params["delay_mix"] > 0.1:
                effects.append(
                    Delay(
                        delay_seconds=params["delay_ms"] / 1000,
                        feedback=params["delay_fb"] / 100,
                        mix=params["delay_mix"] / 100,
                    )
                )
            if params["reverb_wet"] > 0.1:
                wet = params["reverb_wet"] / 100
                effects.append(
                    Reverb(
                        room_size=params["reverb_room"],
                        wet_level=wet,
                        dry_level=1 - wet,
                    )
                )
            if params["comp_ratio"] > 1.0 and params["comp_thresh"] < 0:
                effects.append(
                    Compressor(
                        threshold_db=params["comp_thresh"],
                        ratio=params["comp_ratio"],
                        attack_ms=params["comp_attack"],
                        release_ms=params["comp_release"],
                    )
                )
            if abs(params["gain_db"]) > 0.1:
                effects.append(Gain(gain_db=params["gain_db"]))

            if effects:
                board = Pedalboard(effects)
                if audio.ndim == 2:
                    audio = board(audio.T.astype(np.float32), sr).T.astype(np.float64)
                else:
                    audio = board(audio[np.newaxis, :].astype(np.float32), sr)[0].astype(np.float64)

            audio = audio * params["volume"]
            processed[name] = audio
            if status_callback is not None:
                status_callback(f"処理中... {name} 完了")

        min_len = min(audio.shape[0] for audio in processed.values())
        mix_audio = sum(audio[:min_len] for audio in processed.values())
        master = master_params
        master_effects = []
        if master["m_reverb_wet"] > 0.1:
            wet = master["m_reverb_wet"] / 100
            master_effects.append(
                Reverb(
                    room_size=master["m_reverb_room"],
                    wet_level=wet,
                    dry_level=1 - wet,
                    width=1.0,
                )
            )
        master_effects.append(Limiter(threshold_db=master["m_limiter_db"], release_ms=80))
        board = Pedalboard(master_effects)
        if mix_audio.ndim == 2:
            return board(mix_audio.T.astype(np.float32), sr).T.astype(np.float64)
        return board(mix_audio[np.newaxis, :].astype(np.float32), sr)[0].astype(np.float64)

    def _get_muted_stems(self) -> set:
        return {name for name, cb in self.mute_checks.items() if cb.isChecked()}

    def _preview(self) -> None:
        if not self.stems_raw:
            QMessageBox.warning(self, "警告", "先にステムを読み込んでください")
            return

        sd.stop()
        self.playing = True
        self._cancel = False
        offset = self.preview_offset.value()
        muted = self._get_muted_stems()
        self._set_status(f"プレビュー処理中... ({offset}秒から)")
        self._set_action_buttons_enabled(False)
        self.stop_button.setEnabled(True)
        stem_params, master_params = self._collect_params()

        def task(signals: TaskSignals):
            audio = self._process_stems(
                preview_seconds=30,
                preview_offset=offset,
                stem_params=stem_params,
                master_params=master_params,
                status_callback=signals.status.emit,
                mute_stems=muted,
            )
            if audio is None or self._cancel:
                return
            signals.status.emit("再生中...")
            audio32 = audio.astype(np.float32)
            for attempt in range(3):
                try:
                    sd._terminate()
                    sd._initialize()
                    sd.play(audio32, self.sr)
                    sd.wait()
                    break
                except sd.PortAudioError:
                    if attempt < 2:
                        import time
                        time.sleep(0.5)
                        signals.status.emit("オーディオデバイス再接続中...")
                    else:
                        signals.status.emit("エラー: オーディオデバイスに接続できません")
            if self.playing and not self._cancel:
                signals.status.emit("再生完了")

        def on_finished() -> None:
            self.playing = False
            self._set_action_buttons_enabled(True)

        self._run_task(task, on_finished=on_finished)

    def _stop_preview(self) -> None:
        self._cancel = True   # 処理中でも中断
        sd.stop()             # 再生中なら停止
        self.playing = False
        self.stop_button.setEnabled(False)
        self._set_status("停止")

    def _export(self) -> None:
        if not self.stems_raw:
            QMessageBox.warning(self, "警告", "先にステムを読み込んでください")
            return
        self._cancel = False

        fmt = self.export_format
        default_name = f"{self.track_name}_remix.{fmt}" if self.track_name else f"remix_output.{fmt}"
        default_dir = str(Path(__file__).parent / f"remix_{self.track_name}") if self.track_name else ""
        outpath, _ = QFileDialog.getSaveFileName(
            self,
            "書き出し先",
            str(Path(default_dir) / default_name) if default_dir else default_name,
            f"{fmt.upper()} (*.{fmt});;All Files (*)",
        )
        self._restore_focus()
        if not outpath:
            return
        outpath = self._normalize_output_path(outpath, fmt)

        self._set_status("書き出し中...")
        self._set_action_buttons_enabled(False)
        stem_params, master_params = self._collect_params()
        muted = self._get_muted_stems()

        def task(signals: TaskSignals):
            audio = self._process_stems(
                stem_params=stem_params,
                master_params=master_params,
                status_callback=signals.status.emit,
                mute_stems=muted,
            )
            if audio is None:
                return None
            if fmt == "wav":
                sf.write(outpath, audio, self.sr, subtype="PCM_24")
            else:
                tmp = outpath + ".tmp.wav"
                sf.write(tmp, audio, self.sr, subtype="PCM_24")
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", tmp, "-codec:a", "libmp3lame", "-b:a", "320k", outpath],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    wav_path = Path(outpath).with_suffix(".wav")
                    Path(tmp).rename(wav_path)
                    return ("fallback_wav", str(wav_path))
            return ("ok", outpath)

        def on_result(result) -> None:
            if result is None:
                return
            status, path = result
            if status == "fallback_wav":
                self._set_status(f"ffmpegなし -> WAV保存: {path}")
                QMessageBox.information(self, "完了", f"ffmpeg がないため WAV として保存しました。\n{path}")
                return
            self._set_status(f"書き出し完了: {path}")
            QMessageBox.information(self, "完了", f"書き出し完了:\n{path}")

        self._run_task(task, on_result=on_result, on_finished=lambda: self._set_action_buttons_enabled(True))

    def _export_no_vocals(self) -> None:
        """ボーカルなしで書き出し"""
        if not self.stems_raw:
            QMessageBox.warning(self, "警告", "先にステムを読み込んでください")
            return
        self._cancel = False

        fmt = self.export_format
        default_name = f"{self.track_name}_novocal.{fmt}" if self.track_name else f"remix_novocal.{fmt}"
        default_dir = str(Path(__file__).parent / f"remix_{self.track_name}") if self.track_name else ""
        outpath, _ = QFileDialog.getSaveFileName(
            self,
            "ボーカルなし書き出し先",
            str(Path(default_dir) / default_name) if default_dir else default_name,
            f"{fmt.upper()} (*.{fmt});;All Files (*)",
        )
        self._restore_focus()
        if not outpath:
            return
        outpath = self._normalize_output_path(outpath, fmt)

        self._set_status("ボーカルなしで書き出し中...")
        self._set_action_buttons_enabled(False)
        stem_params, master_params = self._collect_params()

        def task(signals: TaskSignals):
            audio = self._process_stems(
                stem_params=stem_params,
                master_params=master_params,
                status_callback=signals.status.emit,
                mute_stems={"vocals"},
            )
            if audio is None:
                return None
            if fmt == "wav":
                sf.write(outpath, audio, self.sr, subtype="PCM_24")
            else:
                tmp = outpath + ".tmp.wav"
                sf.write(tmp, audio, self.sr, subtype="PCM_24")
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", tmp, "-codec:a", "libmp3lame", "-b:a", "320k", outpath],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    wav_path = Path(outpath).with_suffix(".wav")
                    Path(tmp).rename(wav_path)
                    return ("fallback_wav", str(wav_path))
            return ("ok", outpath)

        def on_result(result) -> None:
            if result is None:
                return
            status, path = result
            if status == "fallback_wav":
                self._set_status(f"ffmpegなし -> WAV保存: {path}")
                QMessageBox.information(self, "完了", f"ffmpeg がないため WAV として保存しました。\n{path}")
                return
            self._set_status(f"ボーカルなし書き出し完了: {path}")
            QMessageBox.information(self, "完了", f"ボーカルなし書き出し完了:\n{path}")

        self._run_task(task, on_result=on_result, on_finished=lambda: self._set_action_buttons_enabled(True))

    def _run_ddsp_flute(self) -> None:
        if not self.stems_raw or "vocals" not in self.stems_raw:
            QMessageBox.warning(self, "警告", "先にステムを読み込んでください")
            return
        self._cancel = False

        default_name = f"{self.track_name}_ddsp_vst_lead.wav" if self.track_name else "ddsp_vst_lead.wav"
        default_dir = str(Path(__file__).parent / f"remix_{self.track_name}") if self.track_name else str(Path(__file__).parent)
        outpath, _ = QFileDialog.getSaveFileName(
            self,
            "DDSP VST 楽器化出力先",
            str(Path(default_dir) / default_name),
            "WAV (*.wav);;All Files (*)",
        )
        self._restore_focus()
        if not outpath:
            return
        outpath = self._normalize_output_path(outpath, "wav")

        self._set_status("DDSP VST 楽器化を準備中...")
        self._set_action_buttons_enabled(False)
        vocals = self.stems_raw["vocals"]
        source_sr = self.sr
        project_dir = Path(__file__).parent.resolve()
        setup_script = project_dir / "ddsp_setup.py"
        transfer_script = project_dir / "ddsp_flute_transfer.py"
        venv_python = self.ddsp_venv_dir / "bin" / "python"
        model_dir = self.ddsp_models_dir / "solo_flute_ckpt"

        def task(signals: TaskSignals):
            def run_checked(cmd: list[str], step_name: str) -> None:
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
                    tail = "\n".join(combined.splitlines()[-40:])
                    raise RuntimeError(f"{step_name} に失敗しました。\n{tail.strip()}")

            with tempfile.TemporaryDirectory(prefix="ddsp_gui_") as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                input_wav = tmp_dir_path / "vocals.wav"
                sf.write(str(input_wav), vocals, source_sr, subtype="PCM_24")

                signals.status.emit("DDSP環境をセットアップ中...")
                signals.progress.emit(10)
                run_checked(
                    [
                        sys.executable,
                        str(setup_script),
                        "--venv",
                        str(self.ddsp_venv_dir),
                        "--models-dir",
                        str(self.ddsp_models_dir),
                        "--model",
                        "Flute",
                    ],
                    "DDSPセットアップ",
                )

                if not venv_python.exists():
                    raise RuntimeError(f"DDSP用Pythonが見つかりません: {venv_python}")

                signals.status.emit("DDSP VST 楽器推論中...")
                signals.progress.emit(40)
                run_checked(
                    [
                        str(venv_python),
                        str(transfer_script),
                        "--input",
                        str(input_wav),
                        "--output",
                        str(outpath),
                        "--model-dir",
                        str(model_dir),
                        "--backend",
                        "auto",
                        "--vst-model",
                        "Flute",
                        "--threshold",
                        "0.7",
                        "--quiet",
                        "36",
                        "--autotune",
                        "0.0",
                        "--pitch-shift",
                        "0.0",
                        "--loudness-shift",
                        "2.5",
                        "--noise-mix",
                        "0.02",
                        "--reverb-mix",
                        "0.0",
                        "--post-lowpass",
                        "4000",
                        "--post-highpass",
                        "90",
                        "--output-drive",
                        "1.0",
                        "--pitch-anchor-mix",
                        "0.18",
                        "--chunk-seconds",
                        "12",
                        "--chunk-overlap-seconds",
                        "0.6",
                    ],
                    "DDSP VST 楽器推論",
                )

                signals.status.emit("DDSP出力をセッションへ反映中...")
                signals.progress.emit(85)
                converted, converted_sr = sf.read(outpath, dtype="float64")
                if converted.ndim == 2:
                    converted = np.mean(converted, axis=1)
                if converted_sr != source_sr:
                    converted = librosa.resample(converted.astype(np.float32), orig_sr=converted_sr, target_sr=source_sr)
                if vocals.ndim == 2:
                    converted = np.column_stack([converted, converted])

                target_len = vocals.shape[0]
                if len(converted) > target_len:
                    converted = converted[:target_len]
                elif len(converted) < target_len:
                    pad_shape = (target_len - len(converted), vocals.shape[1]) if vocals.ndim == 2 else (target_len - len(converted),)
                    converted = np.concatenate([converted, np.zeros(pad_shape, dtype=converted.dtype)], axis=0)

                return np.asarray(converted, dtype=np.float64), outpath

        def on_result(result) -> None:
            if result is None:
                return
            converted, path = result
            self.stems_raw["vocals"] = converted
            self._update_progress(100)
            self._set_status(f"DDSP VST 楽器化完了: VOCALSを置換しました ({path})")
            QMessageBox.information(
                self,
                "完了",
                f"DDSP VST 楽器化が完了しました。\nVOCALSステムを置換し、WAVも保存しました。\n{path}",
            )

        self._run_task(task, on_result=on_result, on_finished=lambda: self._set_action_buttons_enabled(True))

    def _refresh_preset_combo(self) -> None:
        """presetsフォルダをスキャンしてプルダウンを更新"""
        self.preset_combo.blockSignals(True)
        current = self.preset_combo.currentText()
        self.preset_combo.clear()
        self.preset_combo.addItem("-- 選択 --")
        for f in sorted(self.presets_dir.glob("*.json")):
            self.preset_combo.addItem(f.stem)
        # 元の選択を復元
        idx = self.preset_combo.findText(current)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, index: int) -> None:
        """プルダウンでプリセットを選択した時"""
        name = self.preset_combo.currentText()
        if name == "-- 選択 --" or not name:
            return
        filepath = self.presets_dir / f"{name}.json"
        if not filepath.exists():
            return
        self._apply_preset_file(filepath)

    def _save_preset(self) -> None:
        """現在のスライダー設定に名前をつけてJSON保存"""
        name, ok = QInputDialog.getText(self, "プリセット保存", "プリセット名:")
        if not ok or not name.strip():
            return
        name = name.strip()

        preset = {
            "name": name,
            "stems": {},
            "master": {k: ctrl.value() for k, ctrl in self.master_sliders.items()},
            "mute": {n: cb.isChecked() for n, cb in self.mute_checks.items()},
            "preview_offset": self.preview_offset.value(),
            "export_format": self.export_format,
        }
        for sname in self.STEM_NAMES:
            preset["stems"][sname] = self.stem_controls[sname].get_params()

        outpath = self.presets_dir / f"{name}.json"
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(preset, f, indent=2, ensure_ascii=False)

        self._refresh_preset_combo()
        idx = self.preset_combo.findText(name)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self._set_status(f"プリセット保存完了: {name}")

    def _delete_preset(self) -> None:
        """選択中のプリセットを削除"""
        name = self.preset_combo.currentText()
        if name == "-- 選択 --" or not name:
            return
        filepath = self.presets_dir / f"{name}.json"
        if not filepath.exists():
            return
        reply = QMessageBox.question(
            self, "確認", f"プリセット「{name}」を削除しますか？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        filepath.unlink()
        self._refresh_preset_combo()
        self._set_status(f"プリセット削除: {name}")

    def _apply_preset_file(self, filepath: Path) -> None:
        """JSONプリセットファイルを適用"""
        with open(filepath, "r", encoding="utf-8") as f:
            preset = json.load(f)

        for name, params in preset.get("stems", {}).items():
            if name in self.stem_controls:
                self.stem_controls[name].set_params(params)

        for key, value in preset.get("master", {}).items():
            if key in self.master_sliders:
                self.master_sliders[key].set_value(value)

        for name, checked in preset.get("mute", {}).items():
            if name in self.mute_checks:
                self.mute_checks[name].setChecked(checked)

        if "preview_offset" in preset:
            self.preview_offset.setValue(preset["preview_offset"])

        if "export_format" in preset:
            self.export_format = preset["export_format"]

        preset_name = preset.get("name", filepath.stem)
        self._set_status(f"プリセット適用: {preset_name}")

    def _load_v2_preset(self) -> None:
        presets = {
            "drums": dict(
                volume=2.2,
                dist_db=5,
                lowpass_hz=20000,
                delay_ms=120,
                delay_fb=20,
                delay_mix=15,
                reverb_room=0,
                reverb_wet=0,
                comp_thresh=-30,
                comp_ratio=20,
                comp_attack=0.5,
                comp_release=30,
                gain_db=8,
            ),
            "bass": dict(
                volume=1.0,
                dist_db=5,
                lowpass_hz=800,
                delay_ms=100,
                delay_fb=20,
                delay_mix=18,
                reverb_room=0,
                reverb_wet=0,
                comp_thresh=-28,
                comp_ratio=15,
                comp_attack=2,
                comp_release=40,
                gain_db=6,
            ),
            "vocals": dict(
                volume=1.2,
                formant=-4.0,
                dist_db=2,
                lowpass_hz=20000,
                delay_ms=150,
                delay_fb=20,
                delay_mix=15,
                reverb_room=0.65,
                reverb_wet=45,
                comp_thresh=-28,
                comp_ratio=12,
                comp_attack=2,
                comp_release=50,
                gain_db=8,
            ),
            "other": dict(
                volume=0.7,
                dist_db=20,
                lowpass_hz=20000,
                delay_ms=180,
                delay_fb=35,
                delay_mix=30,
                reverb_room=0.7,
                reverb_wet=45,
                comp_thresh=-28,
                comp_ratio=15,
                comp_attack=2,
                comp_release=40,
                gain_db=6,
            ),
        }
        for name, params in presets.items():
            self.stem_controls[name].set_params(params)
        self.master_sliders["m_reverb_room"].set_value(0.4)
        self.master_sliders["m_reverb_wet"].set_value(25)
        self.master_sliders["m_limiter_db"].set_value(-0.1)
        self._set_status("V2プリセット読み込み完了")

    def _load_instrument_preset(self) -> None:
        presets = {
            "drums": dict(
                volume=1.0,
                dist_db=0,
                bitcrush=0,
                clipping_db=0,
                ladder_hz=0,
                lowpass_hz=18000,
                delay_ms=100,
                delay_fb=10,
                delay_mix=0,
                reverb_room=0.0,
                reverb_wet=0,
                comp_thresh=-18,
                comp_ratio=4.0,
                comp_attack=5,
                comp_release=50,
                gain_db=0,
            ),
            "bass": dict(
                volume=1.0,
                dist_db=2,
                bitcrush=0,
                clipping_db=0,
                ladder_hz=0,
                lowpass_hz=3500,
                delay_ms=100,
                delay_fb=10,
                delay_mix=0,
                reverb_room=0.0,
                reverb_wet=0,
                comp_thresh=-20,
                comp_ratio=4.0,
                comp_attack=5,
                comp_release=50,
                gain_db=1,
            ),
            "vocals": dict(
                volume=1.0,
                formant=-4.0,
                instrumentize=82,
                breath_reduce=88,
                tone_darken=55,
                consonant_suppress=82,
                modulation_blur=68,
                grit_drive=78,
                robot_mod=52,
                dist_db=0,
                bitcrush=10,
                clipping_db=0,
                ladder_hz=0,
                lowpass_hz=3000,
                delay_ms=90,
                delay_fb=8,
                delay_mix=0,
                reverb_room=0.15,
                reverb_wet=5,
                comp_thresh=-22,
                comp_ratio=5.0,
                comp_attack=3,
                comp_release=45,
                gain_db=1,
            ),
            "other": dict(
                volume=1.0,
                dist_db=0,
                bitcrush=0,
                clipping_db=0,
                ladder_hz=0,
                lowpass_hz=18000,
                delay_ms=100,
                delay_fb=10,
                delay_mix=0,
                reverb_room=0.0,
                reverb_wet=0,
                comp_thresh=-18,
                comp_ratio=4.0,
                comp_attack=5,
                comp_release=50,
                gain_db=0,
            ),
        }
        for name, params in presets.items():
            self.stem_controls[name].set_params(params)
        self.master_sliders["m_reverb_room"].set_value(0.2)
        self.master_sliders["m_reverb_wet"].set_value(8)
        self.master_sliders["m_limiter_db"].set_value(-0.3)
        self._set_status("歌メロ楽器化プリセット読み込み完了")

    def _reset_all(self) -> None:
        for name in self.STEM_NAMES:
            control = self.stem_controls[name]
            for _, key, _, _, default, _ in control._defaults():
                control.controls[key].set_value(default)
        self.master_sliders["m_reverb_room"].set_value(0.4)
        self.master_sliders["m_reverb_wet"].set_value(25)
        self.master_sliders["m_limiter_db"].set_value(-0.1)
        self._set_status("リセット完了")

    def closeEvent(self, event) -> None:  # noqa: N802
        sd.stop()
        self.playing = False
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = RemixGUI()
    window.show()
    window.activateWindow()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
