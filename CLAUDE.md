# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

FACodec（Factorized Audio Codec）は、NaturalSpeech 3の中核コンポーネントです。音声波形を分離された部分空間（コンテンツ、プロソディ、音色、残差音響詳細）にエンコードし、高品質な音声を再構成します。

- **元プロジェクト**: Amphion (open-mmlab/Amphion)
- **ライセンス**: MIT
- **オーディオ仕様**: 16kHz、ホップサイズ200サンプル

## セットアップとコマンド

```bash
# インストール
pip3 install torch==2.1.2 torchaudio==2.1.2
pip3 install .           # 通常インストール
pip3 install -e .        # 開発モード

# テスト実行（HuggingFaceから事前学習モデルをダウンロード）
python test.py
```

## コードアーキテクチャ

### ns3_codec/ - メインパッケージ

**facodec.py** - 中核のエンコーダ/デコーダ（約1,200行）
- `FACodecEncoder` / `FACodecDecoder`: 標準的なエンコード/デコード
- `FACodecEncoderV2` / `FACodecDecoderV2`: ゼロショット音声変換用の拡張版
- `FACodecRedecoder`: 話者変換用リデコーダ

**quantize/** - ベクトル量子化
- `rvq.py`: 残差VQ（ResidualVQ）
- `fvq.py`: 有限VQ（FVQ）

**alias_free_torch/** - エイリアスフリー処理
- カスタム活性化関数とリサンプリング

**その他**
- `transformer.py`: シーケンス処理用Transformer
- `melspec.py`: メルスペクトログラム計算
- `gradient_reversal.py`: 敵対的学習用勾配反転

### 量子化構造

FACodecDecoderは3段階のVQを使用:
1. **プロソディコード**: 1量子化器
2. **コンテンツコード**: 2量子化器
3. **残差コード**: 3量子化器（音響詳細）

### 主要パターン

- **テンソル形式**: `[B, C, T]`（バッチ、チャンネル、時間）
- **推論モード**: `model.eval()` + `torch.no_grad()` を使用
- **重み正規化**: Conv1d層に適用
- **Snake活性化**: 学習可能なalpha/betaパラメータ

### 事前学習モデル（HuggingFace: amphion/naturalspeech3_facodec）

- `ns3_facodec_encoder.bin` / `ns3_facodec_decoder.bin`: 標準版
- `ns3_facodec_encoder_v2.bin` / `ns3_facodec_decoder_v2.bin`: V2版（音声変換用）
- `ns3_facodec_redecoder.bin`: 話者変換用

## 使用例

`test.py` を参照してください。以下の機能のデモが含まれています:
- オーディオのエンコード/デコード
- VQによる量子化
- 話者埋め込みの抽出
- ゼロショット音声変換
