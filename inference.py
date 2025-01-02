import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize
from scipy.io.wavfile import write

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model paths (You need to change the paths)
xtts_checkpoint = "checkpoints/GPT_XTTS_FT-January-1-2025_08+19AM-6a6b942/best_model_99875.pth"
xtts_config = "checkpoints/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/config.json"
xtts_vocab = "checkpoints/XTTS_v2.0_original_model_files/vocab.json"

# Load model
config = XttsConfig()
config.load_json(xtts_config)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
XTTS_MODEL.to(device)

print("Model loaded successfully!")

# Inference
tts_text="كَما اعْتَبَرَتْ مُنَظَمَةُ الْعَفْوِ الدَّوليَّةُ الْقَرارَ الْإِسْرائِيلِي غَيرَ قَانُونِيٍّ - وَيَجِبُ إِلْغاؤُهُ فَوراً"
speaker_audio_file = "ref.mp3"
lang = "ar"

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)

tts_texts = sent_tokenize(tts_text)

wav_chunks = []
for text in tqdm(tts_texts):
    wav_chunk = XTTS_MODEL.inference(
        text=text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.1,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=10,
        top_p=0.3,
    )
    wav_chunks.append(torch.tensor(wav_chunk["wav"]))

out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()

# Assuming a sample rate (you can adjust this based on your model's output)
sample_rate = 24000  # Adjust if necessary

# Convert tensor to numpy array
out_wav_numpy = out_wav.squeeze(0).numpy()

# Save as a WAV file
output_file = "output_audio.wav"
write(output_file, sample_rate, out_wav_numpy)

print(f"Audio saved as {output_file}")