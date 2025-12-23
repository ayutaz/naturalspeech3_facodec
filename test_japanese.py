import librosa
import soundfile as sf
import torch
from ns3_codec import FACodecEncoderV2, FACodecDecoderV2
from huggingface_hub import hf_hub_download

# つくよみちゃんコーパスのパス
CORPUS_PATH = "/Users/s19447/Downloads/つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠）/02 WAV（+12dB増幅）"


def load_audio(wav_path, max_seconds=5.0):
    """音声を読み込み、指定秒数に切り取る"""
    wav = librosa.load(wav_path, sr=16000)[0]
    # 最大秒数に制限
    max_samples = int(max_seconds * 16000)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    wav = torch.from_numpy(wav).float()
    wav = wav.unsqueeze(0).unsqueeze(0)
    return wav


# EncoderV2/DecoderV2をロード（音声変換用）
fa_encoder_v2 = FACodecEncoderV2(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder_v2 = FACodecDecoderV2(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

encoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin")
decoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin")

fa_encoder_v2.load_state_dict(torch.load(encoder_v2_ckpt))
fa_decoder_v2.load_state_dict(torch.load(decoder_v2_ckpt))

fa_encoder_v2.eval()
fa_decoder_v2.eval()

print("=== 日本語音声での音声変換テスト ===\n")

with torch.no_grad():
    # 2つの異なる発話を使用（同じ話者だが異なる内容）
    wav_a_path = f"{CORPUS_PATH}/VOICEACTRESS100_001.wav"
    wav_b_path = f"{CORPUS_PATH}/VOICEACTRESS100_050.wav"

    print(f"音声A（発話内容）: {wav_a_path}")
    print(f"音声B（話者特徴）: {wav_b_path}")

    wav_a = load_audio(wav_a_path)
    wav_b = load_audio(wav_b_path)

    print(f"\n音声A shape: {wav_a.shape}")
    print(f"音声B shape: {wav_b.shape}")

    # エンコード
    enc_out_a = fa_encoder_v2(wav_a)
    prosody_a = fa_encoder_v2.get_prosody_feature(wav_a)
    enc_out_b = fa_encoder_v2(wav_b)
    prosody_b = fa_encoder_v2.get_prosody_feature(wav_b)

    # 量子化
    vq_post_emb_a, vq_id_a, _, quantized, spk_embs_a = fa_decoder_v2(
        enc_out_a, prosody_a, eval_vq=False, vq=True
    )
    vq_post_emb_b, vq_id_b, _, quantized, spk_embs_b = fa_decoder_v2(
        enc_out_b, prosody_b, eval_vq=False, vq=True
    )

    print(f"\n話者埋め込みA shape: {spk_embs_a.shape}")
    print(f"話者埋め込みB shape: {spk_embs_b.shape}")

    # 音声Aの内容を音声Bの話者特徴で再合成
    # （注：同じ話者なので、実際の変化は限定的）
    vq_post_emb_a_to_b = fa_decoder_v2.vq2emb(vq_id_a, use_residual=False)
    recon_wav_a_to_b = fa_decoder_v2.inference(vq_post_emb_a_to_b, spk_embs_b)

    # 音声Aの再構成
    recon_wav_a = fa_decoder_v2.inference(vq_post_emb_a, spk_embs_a)

    # 保存
    output_recon = "./audio/japanese_recon.wav"
    output_vc = "./audio/japanese_vc.wav"

    sf.write(output_recon, recon_wav_a[0][0].cpu().numpy(), 16000)
    sf.write(output_vc, recon_wav_a_to_b[0][0].cpu().numpy(), 16000)

    print(f"\n=== 出力ファイル ===")
    print(f"再構成音声: {output_recon}")
    print(f"音声変換: {output_vc}")
    print(f"\n注: つくよみちゃんコーパスは単一話者のため、")
    print(f"    音声変換の効果は限定的です。")
    print(f"    異なる話者の音声で試すと、より明確な変換が確認できます。")
