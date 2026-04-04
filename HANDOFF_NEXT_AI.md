# 次AI向け引き継ぎ書

最終更新: 2026-04-04 (v2マージ完了)

---

## 1. ユーザーの現在の要求

- 「歌メロ楽器化」で、**高音域がまだ潰れる問題を改善したい**
- 方向性: 「楽器っぽさを増やす」より「**発音だけ減らして、元メロディを最大限保つ**」
- 「高音で潰れないことを優先する。楽器っぽさが少し減ってもよい」


## 2. 開発の現在地

### 本命: `music_remix.py` の `instrumentize_vocal()`

- 旧来の `WORLD` 再合成は廃止済み
- 現在は `STFT + HPSS + 高域抑制 + 原音再注入` ベース
- `DDSP VST 楽器化` は音色は良いが、メロディ保持で不満が残り、現在は主戦場でない

### v2 高音保護改善（マージ完了: 2026-04-04）

セクション3の6変更を `music_remix.py` 本体にマージ済み。合成信号テスト結果:
- src  seg_high4k=0.5572, seg_cent=7843.5
- v2   seg_high4k=0.5158, seg_cent=7302.1 (高域保持しつつ発音感を低減)

### 今回のセッションで判明した重大バグ（修正済み）

`instrumentize_vocal()` の `out_spec` 計算に **OOM クラッシュバグ** があった。

```python
# 旧: harmonic_gain は (n_freqs, n_frames) の 2D なのに
(harmonic * harmonic_gain[:, np.newaxis])
# → (n_freqs, n_freqs, n_frames) という 3D 巨大配列を生成 → RAM 枯渇

# 修正後 (music_remix.py:475)
(harmonic * harmonic_gain)
```

**コミット `94e5180` で修正済み・push 済み。**  
これにより、以前は `instrumentize_vocal()` を実行するたびに裏でメモリを大量消費していた。


## 3. 高音保護改善 (v2) — マージ完了

以下の 6 点を `music_remix.py` 本体にマージ済み (コミット: 次コミット参照)。

| # | 変更内容 | 旧値 | 新値 |
|---|---|---|---|
| 1 | `high_pitch_guard` 発動しきい値 | 205 Hz | 180 Hz |
| 1 | 同・ランプ幅 | 60 Hz | 40 Hz |
| 2 | **スペクトル重心フォールバック追加** | なし | pyin が f0=0 を返す高音フレームも重心で保護 |
| 3 | `high_pitch_relief` 係数 | 0.96 | 0.98 |
| 4 | `dynamic_amount` 削減係数 | 0.78 | 0.85 |
| 4 | `dynamic_breath` 削減係数 | 0.84 | 0.90 |
| 4 | `dynamic_consonant` 削減係数 | 0.82 | 0.88 |
| 4 | `dynamic_blur` 削減係数 | 0.88 | 0.92 |
| 4 | `dynamic_tone` 削減係数 | 0.94 | 0.97 |
| 5 | `bandlimit_high_melody_core` LP | 4200 Hz | 5500 Hz |
| 6 | `conservative_mix` 係数 | 0.50 + 0.32×amount | 0.62 + 0.28×amount |
| 6 | `conservative_mix` 上限 | 0.88 | 0.96 |

### Colab 検証結果（合成ボーカル信号）

| テスト信号 | 高域保存 original | 高域保存 v2 | 改善 |
|---|---|---|---|
| A4 (440Hz) 中音 | 0.666 | 0.683 | +2.6% |
| E5 (659Hz) 高音 | 0.657 | 0.684 | +4.1% |
| A5 (880Hz) 超高音 | 0.646 | 0.678 | +5.0% |

低域・中域への影響ほぼなし。


## 4. 重要コードの場所

### `music_remix.py`

| 関数 / 処理 | 行番号 |
|---|---|
| `bandlimit_melody_core()` | 179 |
| `bandlimit_high_melody_core()` | 189 |
| `soft_hpss()` | 199 |
| `instrumentize_vocal()` 定義 | 270 |
| `high_pitch_guard` 計算 | 411–420 |
| `harmonic_protect` 生成 | 426–441 |
| `dynamic_*` 係数群 | 445–449 |
| **OOM バグ修正箇所** | 474–478 |
| `conservative_mix` (高音 dry blend) | 503–504 |

### `remix_gui.py`

| 処理 | 行番号 |
|---|---|
| `stems_original` 保持 | 230 |
| `instrumentize_vocal()` 呼び出し | 631 |
| `歌メロ楽器化` プリセット | 1228 |

現在の GUI プリセット:
```python
instrumentize=58, breath_reduce=38, tone_darken=10,
consonant_suppress=20, modulation_blur=8,
lowpass_hz=6500, reverb_room=0.02, reverb_wet=0
```


## 5. 未解決の問題

### 5.1 高音保護がまだ不十分な可能性

1. `pyin` の f0 未検出フレームで `high_pitch_guard = 0` になる  
   → v2 でスペクトル重心フォールバックを追加したが、まだ未マージ

2. `tone_darken` と `breath_reduction` が高音時でもまだ残る  
   → v2 で削減係数を強化した

3. `soft_hpss()` の kernel / margin が高音フレーズに攻撃的すぎる可能性  
   → 未対処

4. `sample_keep` / `silence_gate` が高音 sustain を十分保持できていない可能性  
   → 未対処

### 5.2 GUI の制約

- `DDSP VST 楽器化` → `歌メロ楽器化` の順に押すと、元ボーカルでなく差し替え済みボーカルに処理が掛かる
- 「元 VOCALS に戻す」ボタンが GUI にない（`stems_original` は保持済み）


## 6. 次にやるべきこと（優先順）

1. ~~**v2 の変更を `music_remix.py` 本体にマージする**~~ ✅ 完了

2. 実音源 `/tmp/ddsp_backnumber_45_53.wav` の `4.19s–5.63s` 区間で聴感検証  
   数値指標: `4kHz 以上の比率`, `spectral centroid`, `seg_rms`

3. さらに高音潰れが残る場合:
   - `soft_hpss()` の `harmonic_margin` を高音フレームで動的に下げる
   - `sample_keep` を高音時に床上げする

4. GUI に「元 VOCALS へ戻す」ボタンを追加（`stems_original["vocals"]` を使う）


## 7. やらないほうがよいこと

- DDSP 系へ戻す（いまの不満は `歌メロ楽器化` の詰めで解くほうが近い）
- 数値改善だけで終わる（この件は聴感上の不満が中心）
- 高音潰れの再現確認なしで大きく設計変更する


## 8. Git 状態

- リポジトリ: `/Users/gon/Documents/AI_Generated_Apps/music-remix-app`
- ブランチ: `main`
- 最新コミット: v2マージコミット (2026-04-04)
- 主な変更ファイル: `music_remix.py`, `HANDOFF_NEXT_AI.md`, `NEXT_AI_PROMPT.md`


## 9. 検証コマンド（参考）

```bash
# 構文確認
python3 -m py_compile music_remix.py remix_gui.py

# 短尺テスト実行
python3 - <<'PY'
import soundfile as sf
from music_remix import instrumentize_vocal

src = '/tmp/ddsp_backnumber_45_53.wav'
out = '/tmp/instrumentize_v2_test.wav'
audio, sr = sf.read(src, always_2d=False)
res = instrumentize_vocal(
    audio, sr,
    amount=0.58, breath_reduction=0.38, tone_darken=0.10,
    consonant_suppress=0.20, modulation_blur=0.08,
)
sf.write(out, res, sr)
print(out)
PY
```

```bash
# 高音区間の簡易スペクトル評価 (4.19s–5.63s)
python3 - <<'PY'
import numpy as np, soundfile as sf, librosa

def analyze(path, label):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    S = np.abs(librosa.stft(y.astype(np.float32)))
    freqs = librosa.fft_frequencies(sr=sr)
    mag = S.mean(axis=1)
    total = mag.sum() + 1e-8
    h4k = mag[freqs >= 4000].sum() / total
    l220 = mag[freqs <= 220].sum() / total
    cent = (freqs * mag).sum() / total
    # 高音区間
    start, end = int(4.19 * sr), int(5.63 * sr)
    ys = y[start:end]
    Ss = np.abs(librosa.stft(ys.astype(np.float32)))
    ms = Ss.mean(axis=1); ts = ms.sum() + 1e-8
    print(f"{label:8s} global_high4k={h4k:.4f} global_low220={l220:.4f} global_cent={cent:.1f} "
          f"seg_high4k={ms[freqs>=4000].sum()/ts:.4f} seg_cent={(freqs*ms).sum()/ts:.1f} seg_rms={np.sqrt((ys**2).mean()):.4f}")

for path, label in [
    ('/tmp/ddsp_backnumber_45_53.wav', 'src'),
    ('/tmp/instrumentize_v2_test.wav', 'v2'),
]:
    try: analyze(path, label)
    except: print(f"{label}: file not found")
PY
```


## 10. Colab での継続方法

Colab MCP ツールが使える環境なら:

```
mcp__colab-mcp__open_colab_browser_connection  → 接続
mcp__colab-mcp__get_cells                      → 現状確認
mcp__colab-mcp__run_code_cell                  → セル実行
```

前回セッションのノートブックには以下のセルが残っている:
- Cell 1: ヘルパー関数
- Cell 2: `instrumentize_vocal_original` (OOM バグ修正済み)
- Cell 3: `instrumentize_vocal_v2` (高音保護 v2)
- Cell 4: テスト信号生成
- Cell 5: 比較・評価
- Cell 6: スペクトルグラフ
