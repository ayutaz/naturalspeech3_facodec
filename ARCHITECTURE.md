# FACodec アーキテクチャ

このドキュメントでは、FACodec（NaturalSpeech 3の音声コーデック）の内部アーキテクチャを詳細に解説します。

## 全体構造

```
┌─────────────────────────────────────────────────────────────────┐
│                        FACodec                                  │
├─────────────────────────────────────────────────────────────────┤
│  Audio [B,1,T]                                                  │
│       ↓                                                         │
│  ┌─────────────┐                                                │
│  │ FACodecEncoder │ → 潜在表現 [B,256,T/200]                    │
│  └─────────────┘                                                │
│       ↓                                                         │
│  ┌─────────────┐                                                │
│  │ FACodecDecoder │                                             │
│  │  ├─ VQ(プロソディ) × 1量子化器                               │
│  │  ├─ VQ(コンテンツ) × 2量子化器                               │
│  │  ├─ VQ(残差)     × 3量子化器                                 │
│  │  └─ 話者埋め込み抽出                                         │
│  └─────────────┘                                                │
│       ↓                                                         │
│  再構成音声 [B,1,T]                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. エンコーダ (FACodecEncoder)

### 処理フロー
```
入力: [B, 1, T]
  ↓
WNConv1d(kernel=7) → ngf=32
  ↓
EncoderBlock(stride=2) → 64ch  ← ダウンサンプル2倍
  ↓
EncoderBlock(stride=4) → 128ch ← ダウンサンプル4倍
  ↓
EncoderBlock(stride=5) → 256ch ← ダウンサンプル5倍
  ↓
EncoderBlock(stride=5) → 512ch ← ダウンサンプル5倍
  ↓
SnakeBeta + WNConv1d → 256ch
  ↓
出力: [B, 256, T/200]  (hop_length = 2×4×5×5 = 200)
```

### EncoderBlock構成
```
ResidualUnit(dilation=1) → ResidualUnit(dilation=3) → ResidualUnit(dilation=9)
  ↓
Activation1d(SnakeBeta)  ← エイリアスフリー処理
  ↓
WNConv1d(stride=N)       ← ダウンサンプリング
```

### V2の追加機能
- `MelSpectrogram`: メルスペクトログラム計算（80バンド）
- `get_prosody_feature()`: 最初の20メルバンドからプロソディ抽出

---

## 2. デコーダ (FACodecDecoder)

### 3段階量子化の仕組み

| 段階 | 量子化器数 | 捉える情報 | 敵対的学習で排除 |
|------|-----------|-----------|-----------------|
| プロソディ | 1 | ピッチ、リズム、エネルギー | 音素情報 |
| コンテンツ | 2 | 音素、言語内容 | F0情報 |
| 残差 | 3 | 音響詳細、ノイズ | F0/音素 |

### 量子化処理フロー
```
入力x [B, 256, T]
  ↓
┌─────────────────────────────────────────────┐
│ [VQ0] プロソディ量子化                       │
│   out0, idx0 = quantizer[0](x)              │
├─────────────────────────────────────────────┤
│ [VQ1] コンテンツ量子化                       │
│   out1, idx1 = quantizer[1](x)              │
├─────────────────────────────────────────────┤
│ [VQ2] 残差量子化                             │
│   residual = x - out0.detach() - out1.detach()│
│   out2, idx2 = quantizer[2](residual)       │
└─────────────────────────────────────────────┘
  ↓
quantized = out0 + out1 + out2
```

### 話者埋め込み抽出
```
入力x [B, 256, T]
  ↓
TransformerEncoder(timbre_encoder)
  ↓
時間軸で平均化 (mean over T)
  ↓
spk_embs [B, 256]
```

### 音声再構成（inference）
```
vq_post_emb + spk_embs
  ↓
Style適用: γ, β = timbre_linear(spk_embs)
           x = norm(x) × γ + β
  ↓
DecoderBlock(stride=5) → アップサンプル5倍
DecoderBlock(stride=5) → アップサンプル5倍
DecoderBlock(stride=4) → アップサンプル4倍
DecoderBlock(stride=2) → アップサンプル2倍
  ↓
SnakeBeta + Conv1d + Tanh
  ↓
出力: [B, 1, T]
```

---

## 3. 量子化モジュール

### ResidualVQ (残差ベクトル量子化)
```
入力x
  ↓
for each quantizer:
    q = quantize(residual)
    residual = residual - q  ← 残差更新
    output += q              ← 累積
  ↓
出力: quantized, indices, loss
```

### FactorizedVQ (FVQ)
```
入力z_e
  ↓
in_proj: dim → codebook_dim (8)
  ↓
L2正規化 → 最近傍コードブック検索
  ↓
out_proj: codebook_dim → dim
  ↓
Straight-through: z_q = z_e + (z_q - z_e).detach()
```

---

## 4. 補助モジュール

### SnakeBeta活性化
```
SnakeBeta(x) = x + (1/β) × sin²(α × x)
```
- α, β: 学習可能パラメータ
- 周期性と振幅を制御

### Activation1d（エイリアスフリー）
```
入力 → Upsample(2x) → SnakeBeta → Downsample(2x) → 出力
```
- 非線形処理による高周波歪みを防止

### GradientReversal（敵対的学習）
```
Forward:  x → x（恒等写像）
Backward: grad → -α × grad（勾配反転）
```
- 特定の情報を「学習しない」よう強制
- 例: コンテンツコードからF0を予測不可に

### TransformerEncoder
- 4層、256次元、4ヘッド
- StyleAdaptiveLayerNorm: 条件付き正規化
- 話者埋め込み抽出に使用

---

## 5. V1 vs V2 の違い

| 機能 | V1 | V2 |
|------|----|----|
| プロソディ抽出 | VQから直接 | メルスペクトログラムから |
| 音声変換 | 基本的 | 高品質（prosody_feature入力） |
| forward引数 | x のみ | x, prosody_feature |

---

## 6. 重要なパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| ngf | 32 | 基本チャンネル数 |
| up_ratios | [2,4,5,5] | アップ/ダウンサンプル比 |
| hop_length | 200 | 2×4×5×5 |
| vq_dim | 256 | 量子化特徴次元 |
| codebook_dim | 8 | コードブック埋め込み次元 |
| codebook_size | 2^10=1024 | コード語数 |
| sample_rate | 16kHz | サンプリングレート |

---

## 7. 音声変換の仕組み

```python
# 話者Aの音声からコンテンツを抽出
enc_a = encoder(wav_a)
vq_post_a, vq_id_a, _, _, spk_embs_a = decoder(enc_a, vq=True)

# 話者Bの音声から話者埋め込みを抽出
enc_b = encoder(wav_b)
_, _, _, _, spk_embs_b = decoder(enc_b, vq=True)

# Aのコンテンツ + Bの話者 = Bの声でAの内容
vq_emb = decoder.vq2emb(vq_id_a, use_residual=False)
output = decoder.inference(vq_emb, spk_embs_b)
```

`use_residual=False`: 残差コードを使わない（話者特有の音響詳細を除去）

---

## 主要ファイル

| ファイル | 行数 | 内容 |
|----------|------|------|
| ns3_codec/facodec.py | ~1200 | エンコーダ/デコーダ本体 |
| ns3_codec/quantize/rvq.py | ~90 | ResidualVQ |
| ns3_codec/quantize/fvq.py | ~120 | FactorizedVQ |
| ns3_codec/transformer.py | ~230 | Transformer |
| ns3_codec/melspec.py | ~100 | メルスペクトログラム |
| ns3_codec/gradient_reversal.py | ~35 | 勾配反転 |
| ns3_codec/alias_free_torch/ | ~180 | エイリアスフリー処理 |

---

## 詳細1: 話者埋め込み（Speaker Embedding）

### timbre_encoder の構成
```
TransformerEncoder:
  - 層数: 4
  - 隠れ次元: 256
  - ヘッド数: 4
  - FFNフィルタ: 1024
  - ドロップアウト: 0.1
```

### 話者埋め込み抽出フロー
```python
# facodec.py 467-471行目
x_timbre = x.transpose(1, 2)           # [B, C, T] → [B, T, C]
x_timbre = self.timbre_encoder(x_timbre)  # Transformer処理
x_timbre = x_timbre.transpose(1, 2)    # [B, T, C] → [B, C, T]
spk_embs = torch.mean(x_timbre, dim=2) # 時間軸平均 → [B, C]
```

### Style Adaptive Layer Normalization
```python
# timbre_linear: 話者埋め込みからγ, βを生成
style = self.timbre_linear(speaker_embedding)  # [B, 256] → [B, 512]
gamma, beta = style.chunk(2, 1)                # 分割

# 適用: 正規化後にアフィン変換
x = self.timbre_norm(x)    # LayerNorm
x = x * gamma + beta       # 話者スタイル注入
```

### inference()での使用
```python
def inference(self, x, speaker_embedding):
    gamma, beta = self.timbre_linear(speaker_embedding).chunk(2, 1)
    x = self.timbre_norm(x) * gamma + beta  # 話者情報適用
    x = self.model(x)  # デコード
    return x
```

---

## 詳細2: 敵対的学習（Gradient Reversal）

### 実装原理
```python
class GradientReversal(Function):
    def forward(ctx, x, alpha):
        return x  # 順伝播: そのまま通過

    def backward(ctx, grad_output):
        return -alpha * grad_output  # 逆伝播: 勾配反転
```

### 5つの敵対的学習設定

| フラグ | 予測器 | 入力層 | 出力 | 目的 |
|--------|--------|--------|------|------|
| use_gr_content_f0 | content_f0_predictor | layer_1 | F0 | コンテンツからF0を排除 |
| use_gr_prosody_phone | prosody_phone_predictor | layer_0 | 音素(5003) | プロソディから音素を排除 |
| use_gr_residual_f0 | res_f0_predictor | layer_2 | F0 | 残差からF0を排除 |
| use_gr_residual_phone | res_phone_predictor | layer_2 | 音素(5003) | 残差から音素を排除 |
| use_gr_x_timbre | x_timbre_predictor | 全体x | 話者(245200) | 統合表現から話者を排除 |

### CNNLSTM予測器の構造
```
ResidualUnit(dilation=1) → ResidualUnit(dilation=2) → ResidualUnit(dilation=3)
    ↓
SnakeBeta活性化
    ↓
Linear(indim → outdim) × head数
```

### 敵対的学習の効果
```
layer_0 ← 正勾配(F0学習) + 反転勾配(音素削除) → F0に特化
layer_1 ← 正勾配(音素学習) + 反転勾配(F0削除) → 音素に特化
layer_2 ← 反転勾配(F0削除) + 反転勾配(音素削除) → 純粋な詳細
```

---

## 詳細3: 量子化（VQ）

### ResidualVQ のカスケード処理
```python
quantized_out = 0.0
residual = x

for layer in self.layers:
    quantized = layer(residual)
    residual = residual - quantized  # 残差更新
    quantized_out += quantized       # 累積
```

### FactorizedVQ の処理フロー
```
入力 x [B, vq_dim=1024, T]
    ↓ in_proj (1024 → 8)
z_e [B, codebook_dim=8, T]
    ↓ L2正規化 → コードブック検索（1024個から）
z_q [B, 8, T]
    ↓ out_proj (8 → 1024)
出力 [B, 1024, T]
```

### Straight-through Gradient Estimator
```python
z_q = z_e + (z_q - z_e).detach()
# Forward: z_q を使用
# Backward: z_e に勾配が流れる（z_q - z_e は勾配なし）
```

### 3段階VQの入力
```python
# プロソディVQ: 入力 x をそのまま
out0 = quantizer[0](x)

# コンテンツVQ: 入力 x をそのまま
out1 = quantizer[1](x)

# 残差VQ: プロソディ+コンテンツを引いた残差
residual = x - (out0 + out1).detach()
out2 = quantizer[2](residual)
```

### detach() の意味
- **目的**: 勾配遮断で各VQの独立性を保証
- **効果**: 残差VQがプロソディ/コンテンツの学習を乱さない

### vq2emb() のuse_residualパラメータ
```python
def vq2emb(self, vq, use_residual_code=True):
    out = quantizer[0].vq2emb(prosody_codes)
    out += quantizer[1].vq2emb(content_codes)
    if use_residual_code:
        out += quantizer[2].vq2emb(residual_codes)
    return out
```
- `True`: 高品質再構成（全VQ使用）
- `False`: 音声変換用（話者固有の詳細を除外）

### コードブックパラメータの関係
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| vq_dim | 1024 | エンコーダ出力次元 |
| codebook_dim | 8 | コードベクトル次元（圧縮後） |
| codebook_size | 2^10=1024 | コード語数 |
| 圧縮率 | 1024→8 | 128倍圧縮 |
