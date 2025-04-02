import ntpath
import os
import torch
import argparse
import json
from glob import glob
import numpy as np
import torchaudio
from scipy.io.wavfile import write
import librosa

# Set the torchaudio backend to "soundfile"
torchaudio.set_audio_backend("soundfile")
print("Current torchaudio backend:", torchaudio.get_audio_backend())

# Set random seed for reproducibility
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def load_audio(path):
    # Normalize the file path
    path = os.path.normpath(path)
    audio, sr = torchaudio.load(path)
    audio = audio[:1]  # Use the first channel if there are multiple channels.
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000, resampling_method="kaiser_window")
    # Pad audio so that its length is a multiple of 1280
    p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1]
    audio = torch.nn.functional.pad(audio, (0, p))
    return audio, 16000

def save_audio(wav, out_file, syn_sr=16000):
    wav = (wav.squeeze() / wav.abs().max() * 0.999 * 32767.0).cpu().numpy().astype('int16')
    write(out_file, syn_sr, wav)

def pitch_shift(audio, sample_rate, semitones=3):
    """
    Shifts the pitch of the given audio by the specified number of semitones
    using librosa's pitch_shift function.
    """
    # Convert torch tensor to numpy array (squeeze removes the channel dimension)
    audio_np = audio.squeeze().cpu().numpy()
    # Apply pitch shift using librosa (n_steps=semitones)
    shifted = librosa.effects.pitch_shift(audio_np, sample_rate, n_steps=semitones)
    # Convert back to torch tensor and add a channel dimension back
    shifted = torch.tensor(shifted).unsqueeze(0)
    return shifted

def inference(a):
    print(f'>> Processing {a.src_path}...')
    audio, sr = load_audio(a.src_path)
    # Apply pitch shifting by three semitones
    shifted_audio = pitch_shift(audio, sr, semitones=3)
    save_audio(shifted_audio, a.output_path)
    print(f"Processed and saved: {a.output_path}")

def process_folder(audio_folder):
    participants = sorted(os.listdir(audio_folder))  # Get sorted participant folders
    for participant in participants:
        participant_path = os.path.join(audio_folder, participant)
        if not os.path.isdir(participant_path):
            continue  # Skip if not a directory
        audio_clips = sorted(glob(os.path.join(participant_path, "*.wav")))
        for clip in audio_clips:
            print(f'Processing {clip}...')
            a.src_path = clip
            a.output_path = clip  # Overwrite the original file
            inference(a)
    print(">> Processing Complete.")

def main():
    print('>> Initializing Inference Process...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_folder', type=str, default='./voxceleb_-3_semi_tone', help='Path to folder containing participant audio')
    global a
    a = parser.parse_args()
    process_folder(a.audio_folder)

if __name__ == '__main__':
    main()
