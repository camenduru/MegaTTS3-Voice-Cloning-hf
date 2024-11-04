import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from utils import load_ckpt, print_colored
from tokenizer import make_tokenizer
from model import get_hertz_dev_config
import matplotlib.pyplot as plt

device = 'cuda' if T.cuda.is_available() else 'cpu'
T.cuda.set_device(0)
print_colored(f"Using device: {device}", "grey")

audio_tokenizer = make_tokenizer(device)

TWO_SPEAKER = False

model_config = get_hertz_dev_config(is_split=TWO_SPEAKER)

generator = model_config()
generator = generator.eval().to(T.bfloat16).to(device)



##############
# Load audio

def load_and_preprocess_audio(audio_path):
    gr.Info("Loading and preprocessing audio...")
    # Load audio file
    audio_tensor, sr = torchaudio.load(audio_path)
    gr.Info(f"Loaded audio shape: {audio_tensor.shape}")
    
    if TWO_SPEAKER:
        if audio_tensor.shape[0] == 1:
            gr.Info("Converting mono to stereo...")
            audio_tensor = audio_tensor.repeat(2, 1)
            gr.Info(f"Stereo audio shape: {audio_tensor.shape}")
    else:
        if audio_tensor.shape[0] == 2:
            gr.Info("Converting stereo to mono...")
            audio_tensor = audio_tensor.mean(dim=0).unsqueeze(0)
            gr.Info(f"Mono audio shape: {audio_tensor.shape}")
        
    # Resample to 16kHz if needed
    if sr != 16000:
        gr.Info(f"Resampling from {sr}Hz to 16000Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio_tensor = resampler(audio_tensor)
        
    # Clip to 5 minutes if needed
    max_samples = 16000 * 60 * 5
    if audio_tensor.shape[1] > max_samples:
        # gr.Info("Clipping audio to 5 minutes...")
        raise gr.Erorr("Maximum prompt is 5 minutes")
        # audio_tensor = audio_tensor[:, :max_samples]

    duration_seconds = audio_tensor.shape[1] / sample_rate

    gr.Info("Audio preprocessing complete!")
    return audio_tensor.unsqueeze(0), duration_seconds

##############
# Return audio to gradio

def display_audio(audio_tensor):
    audio_tensor = audio_tensor.cpu().squeeze()
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = audio_tensor.float()

    # Make a waveform plot
    # plt.figure(figsize=(4, 1))
    # plt.plot(audio_tensor.numpy()[0], linewidth=0.5)
    # plt.axis('off')
    # plt.show()

    # Make an audio player
    return (16000, audio_tensor.numpy())

def get_completion(encoded_prompt_audio, prompt_len):
    prompt_len_seconds = prompt_len / 8
    gr.Info(f"Prompt length: {prompt_len_seconds:.2f}s")
    with T.autocast(device_type='cuda', dtype=T.bfloat16):
        completed_audio_batch = generator.completion(
            encoded_prompt_audio, 
            temps=(.8, (0.5, 0.1)), # (token_temp, (categorical_temp, gaussian_temp))
            use_cache=True)

        completed_audio = completed_audio_batch
        print_colored(f"Decoding completion...", "blue")
        if TWO_SPEAKER:
            decoded_completion_ch1 = audio_tokenizer.data_from_latent(completed_audio[:, :, :32].bfloat16())
            decoded_completion_ch2 = audio_tokenizer.data_from_latent(completed_audio[:, :, 32:].bfloat16())
            decoded_completion = T.cat([decoded_completion_ch1, decoded_completion_ch2], dim=0)
        else:
            decoded_completion = audio_tokenizer.data_from_latent(completed_audio.bfloat16())
        gr.Info(f"Decoded completion shape: {decoded_completion.shape}")

    gr.Info("Preparing audio for playback...")

    audio_tensor = decoded_completion.cpu().squeeze()
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = audio_tensor.float()

    if audio_tensor.abs().max() > 1:
        audio_tensor = audio_tensor / audio_tensor.abs().max()

    return audio_tensor[:, max(prompt_len*2000 - 16000, 0):]

@spaces.GPU
def run(audio_path):
    prompt_audio, prompt_len_seconds = load_and_preprocess_audio(audio_path)
    prompt_len = prompt_len_seconds * 8
    gr.Info("Encoding prompt...")
    with T.autocast(device_type='cuda', dtype=T.bfloat16):
        if TWO_SPEAKER:
            encoded_prompt_audio_ch1 = audio_tokenizer.latent_from_data(prompt_audio[:, 0:1].to(device))
            encoded_prompt_audio_ch2 = audio_tokenizer.latent_from_data(prompt_audio[:, 1:2].to(device))
            encoded_prompt_audio = T.cat([encoded_prompt_audio_ch1, encoded_prompt_audio_ch2], dim=-1)
        else:
            encoded_prompt_audio = audio_tokenizer.latent_from_data(prompt_audio.to(device))
    gr.Info(f"Encoded prompt shape: {encoded_prompt_audio.shape}")
    gr.Info("Prompt encoded successfully!")
    # num_completions = 10
    completion = get_completion(encoded_prompt_audio, prompt_len)
    return display_audio(completion)

