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
    max_samples = int(max_seconds * 16000)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    wav = torch.from_numpy(wav).float()
    wav = wav.unsqueeze(0).unsqueeze(0)
    return wav


# EncoderV2/DecoderV2をロード
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

print("=== つくよみちゃんの声で英語を話させる ===\n")

with torch.no_grad():
    # 英語音声（コンテンツ）
    english_wav_path = "./audio/1.wav"
    # つくよみちゃんの音声（話者特徴）
    tsukuyomi_wav_path = f"{CORPUS_PATH}/VOICEACTRESS100_001.wav"

    print(f"英語音声（コンテンツ）: {english_wav_path}")
    print(f"つくよみちゃん音声（話者特徴）: {tsukuyomi_wav_path}")

    wav_english = load_audio(english_wav_path)
    wav_tsukuyomi = load_audio(tsukuyomi_wav_path)

    print(f"\n英語音声 shape: {wav_english.shape}")
    print(f"つくよみちゃん音声 shape: {wav_tsukuyomi.shape}")

    # エンコード
    enc_out_english = fa_encoder_v2(wav_english)
    prosody_english = fa_encoder_v2.get_prosody_feature(wav_english)
    enc_out_tsukuyomi = fa_encoder_v2(wav_tsukuyomi)
    prosody_tsukuyomi = fa_encoder_v2.get_prosody_feature(wav_tsukuyomi)

    # 量子化して話者埋め込みを取得
    vq_post_emb_english, vq_id_english, _, _, spk_embs_english = fa_decoder_v2(
        enc_out_english, prosody_english, eval_vq=False, vq=True
    )
    vq_post_emb_tsukuyomi, vq_id_tsukuyomi, _, _, spk_embs_tsukuyomi = fa_decoder_v2(
        enc_out_tsukuyomi, prosody_tsukuyomi, eval_vq=False, vq=True
    )

    print(f"\n英語話者埋め込み shape: {spk_embs_english.shape}")
    print(f"つくよみちゃん話者埋め込み shape: {spk_embs_tsukuyomi.shape}")

    # 英語のコンテンツ + つくよみちゃんの話者特徴 = つくよみちゃんの声で英語
    vq_post_emb_for_vc = fa_decoder_v2.vq2emb(vq_id_english, use_residual=False)
    recon_wav_tsukuyomi_english = fa_decoder_v2.inference(vq_post_emb_for_vc, spk_embs_tsukuyomi)

    # 比較用：元の英語音声の再構成
    recon_wav_english = fa_decoder_v2.inference(vq_post_emb_english, spk_embs_english)

    # 保存
    output_vc = "./audio/tsukuyomi_english.wav"
    output_original_recon = "./audio/english_recon.wav"

    sf.write(output_vc, recon_wav_tsukuyomi_english[0][0].cpu().numpy(), 16000)
    sf.write(output_original_recon, recon_wav_english[0][0].cpu().numpy(), 16000)

    print(f"\n=== 出力ファイル ===")
    print(f"つくよみちゃんの声で英語: {output_vc}")
    print(f"元の英語音声の再構成: {output_original_recon}")
    print(f"\n聴き比べてみてください！")
