import ntpath
import os
import torch
import argparse
import tqdm
import json
import glob
import numpy as np
import torchaudio
from scipy.io.wavfile import write
import librosa

# Set the torchaudio backend to "soundfile"
torchaudio.set_audio_backend("soundfile")
print("Current torchaudio backend:", torchaudio.get_audio_backend())

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
    # print(f'>> Processing {a.src_path}...')
    audio, sr = load_audio(a.src_path)
    # Apply pitch shifting by three semitones
    shifted_audio = pitch_shift(audio, sr, semitones=a.semitones)
    save_audio(shifted_audio, a.output_path)
    # print(f"Processed and saved: {a.output_path}")

def process_folder(a):

    audio_clips = glob.glob(f"{a.audio_folder}/**/*.wav", recursive=True)

    for clip in tqdm.tqdm(audio_clips):

        folder = ntpath.basename(a.audio_folder)
        anon_folder = f"{folder}_pitchshift_{a.semitones}"

        new_clip = clip.replace(folder, anon_folder)
        os.makedirs(ntpath.dirname(new_clip), exist_ok=True)

        # print(f"Processing {clip} to {new_clip}...")
        a.src_path = clip
        a.output_path = new_clip
        inference(a)

    print(">> Processing Complete.")

def main():
    print('>> Initializing Inference Process...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_folder', type=str, default='./voxceleb', help='Path to folder containing participant audio')
    parser.add_argument('--semitones', type=int, default=3, help='Number of semitones to pitch shift by.')
    global a
    a = parser.parse_args()
    process_folder(a)

if __name__ == '__main__':
    main()
