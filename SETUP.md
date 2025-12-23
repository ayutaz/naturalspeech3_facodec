# FACodec 環境構築・推論ガイド

このドキュメントでは、FACodec（NaturalSpeech 3の音声コーデック）の環境構築から推論までの手順を説明します。

## 目次

1. [環境構築](#環境構築)
2. [基本的な推論](#基本的な推論)
3. [日本語音声でのテスト](#日本語音声でのテスト)
4. [異言語間音声変換](#異言語間音声変換)
5. [FACodecの制限事項](#facodecの制限事項)

---

## 環境構築

### 前提条件

- Python 3.11
- uv（Pythonパッケージマネージャー）

### 手順

#### 1. リポジトリのクローン

```bash
git clone https://github.com/lifeiteng/naturalspeech3_facodec.git
cd naturalspeech3_facodec
```

#### 2. uvプロジェクトの初期化

```bash
uv init --no-readme
```

#### 3. Pythonバージョンの設定

`pyproject.toml`の`requires-python`を`>=3.11`に変更し、Python 3.11を使用するように設定します。

```bash
uv python pin 3.11
```

#### 4. 依存関係のインストール

```bash
# PyTorch（torch 2.1.2が推奨）
uv add torch==2.1.2 torchaudio==2.1.2

# その他の依存関係
uv add pyworld soundfile "librosa==0.10.1" einops huggingface_hub

# NumPyとsetuptoolsの互換性対応
uv add "numpy<2" setuptools
```

#### 5. ns3_codecパッケージのインストール

`setup.py`を`setup.py.bak`にリネームし、`pyproject.toml`でパッケージを管理します。

```bash
mv setup.py setup.py.bak
uv pip install -e .
```

#### 6. pyproject.tomlの設定例

```toml
[project]
name = "ns3-codec"
version = "0.2.2"
description = "FACodec: Speech Codec with Attribute Factorization for NaturalSpeech 3"
requires-python = ">=3.11"
dependencies = [
    "einops>=0.8.1",
    "huggingface-hub>=1.2.3",
    "librosa==0.10.1",
    "numpy<2",
    "pyworld>=0.3.5",
    "setuptools>=80.9.0",
    "soundfile>=0.13.1",
    "torch==2.1.2",
    "torchaudio==2.1.2",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["ns3_codec*"]
```

---

## 基本的な推論

### テストの実行

```bash
uv run python test.py
```

### 出力

- `audio/1_recon.wav`: 再構成された音声
- `audio/1_to_2_vc.wav`: 音声変換結果（話者1→話者2）

### 出力例

```
torch.Size([1, 256, 628])           # エンコーダ出力
vq id shape: torch.Size([6, 1, 628]) # 6つの量子化器
prosody code shape: torch.Size([1, 1, 628])   # プロソディ（1量子化器）
content code shape: torch.Size([2, 1, 628])   # コンテンツ（2量子化器）
residual code shape: torch.Size([3, 1, 628])  # 残差（3量子化器）
speaker embedding shape: torch.Size([1, 256]) # 話者埋め込み
```

### FACodecの処理フロー

```
音声波形 → FACodecEncoder → 潜在表現
                              ↓
                         FACodecDecoder（量子化）
                              ↓
            [プロソディ, コンテンツ, 残差, 話者埋め込み]
                              ↓
                         FACodecDecoder（推論）
                              ↓
                          再構成音声
```

---

## 日本語音声でのテスト

### 使用コーパス

つくよみちゃんコーパス Vol.1（声優統計コーパス）

### テストスクリプト

```python
# test_japanese.py
import librosa
import soundfile as sf
import torch
from ns3_codec import FACodecEncoderV2, FACodecDecoderV2
from huggingface_hub import hf_hub_download

CORPUS_PATH = "/path/to/つくよみちゃんコーパス/02 WAV（+12dB増幅）"

def load_audio(wav_path, max_seconds=5.0):
    """音声を読み込み、指定秒数に切り取る"""
    wav = librosa.load(wav_path, sr=16000)[0]
    max_samples = int(max_seconds * 16000)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    wav = torch.from_numpy(wav).float()
    wav = wav.unsqueeze(0).unsqueeze(0)
    return wav
```

### 実行

```bash
uv run python test_japanese.py
```

### 出力

- `audio/japanese_recon.wav`: 日本語音声の再構成
- `audio/japanese_vc.wav`: 日本語音声変換結果

### 注意点

- 音声は最大5秒に制限（長い音声はテンソルサイズ不一致エラーの原因）
- FACodecは16kHzを期待（librosaが自動変換）

---

## 異言語間音声変換

英語音声のコンテンツを、日本語話者（つくよみちゃん）の声で発話させる。

### 処理フロー

```
英語音声（audio/1.wav）     → コンテンツ抽出（vq_id）
つくよみちゃん音声           → 話者埋め込み抽出（spk_embs）
                              ↓
              fa_decoder_v2.vq2emb(vq_id_english)
              fa_decoder_v2.inference(emb, spk_embs_tsukuyomi)
                              ↓
              つくよみちゃんの声で英語を発話
```

### テストスクリプト

```python
# test_tsukuyomi_english.py
with torch.no_grad():
    # 英語音声からコンテンツを抽出
    enc_out_english = fa_encoder_v2(wav_english)
    prosody_english = fa_encoder_v2.get_prosody_feature(wav_english)
    vq_post_emb_english, vq_id_english, _, _, spk_embs_english = fa_decoder_v2(
        enc_out_english, prosody_english, eval_vq=False, vq=True
    )

    # つくよみちゃんの音声から話者埋め込みを抽出
    enc_out_tsukuyomi = fa_encoder_v2(wav_tsukuyomi)
    prosody_tsukuyomi = fa_encoder_v2.get_prosody_feature(wav_tsukuyomi)
    _, _, _, _, spk_embs_tsukuyomi = fa_decoder_v2(
        enc_out_tsukuyomi, prosody_tsukuyomi, eval_vq=False, vq=True
    )

    # 英語コンテンツ + つくよみちゃんの声 = 変換音声
    vq_post_emb_for_vc = fa_decoder_v2.vq2emb(vq_id_english, use_residual=False)
    recon_wav = fa_decoder_v2.inference(vq_post_emb_for_vc, spk_embs_tsukuyomi)
```

### 実行

```bash
uv run python test_tsukuyomi_english.py
```

### 出力

- `audio/tsukuyomi_english.wav`: つくよみちゃんの声で英語を発話
- `audio/english_recon.wav`: 元の英語音声の再構成（比較用）

---

## FACodecの制限事項

### 1. 話者再現度の限界

FACodecの話者埋め込みは**音色の大まかな特徴**（性別、声の高さなど）を捉えますが、特定の話者の声を完全に再現する設計ではありません。

| 技術 | 目的 | 話者再現度 |
|------|------|------------|
| FACodec | 音声の分離・再構成 | 低〜中 |
| 専用Speaker Encoder | 話者の声を再現 | 高 |
| Voice Cloning | 完全な声の複製 | 非常に高 |

### 2. 音声長の制限

長い音声を処理すると、prosody featureとencoder outputのテンソルサイズが一致しなくなる場合があります。5秒程度に制限することを推奨。

### 3. TTSはできない

FACodecは音声コーデックであり、テキストから音声を生成するTTS機能はありません。TTSには拡散モデルなどの追加コンポーネントが必要です。

```
FACodecでできること:
✓ 音声 → 潜在表現 → 音声（再構成）
✓ 音声A + 音声B → 音声Bの声でAの内容（音声変換）

FACodecでできないこと:
✗ テキスト → 音声（TTS）
```

### 4. 事前学習モデルの依存

推論には事前学習済みモデル（HuggingFace: amphion/naturalspeech3_facodec）が必要です。初回実行時に自動ダウンロードされます。

---

## 参考リンク

- [NaturalSpeech 3 論文](https://arxiv.org/pdf/2403.03100.pdf)
- [HuggingFace モデル](https://huggingface.co/amphion/naturalspeech3_facodec)
- [オリジナルリポジトリ（Amphion）](https://github.com/open-mmlab/Amphion)
