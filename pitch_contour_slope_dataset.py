import os
import glob
import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt

def compute_pitch_variability(audio_path, sr=16000, fmin='C2', fmax='C7'):
    y, sr = librosa.load(audio_path, sr=sr)
    fmin_hz = librosa.note_to_hz(fmin) if isinstance(fmin, str) else fmin
    fmax_hz = librosa.note_to_hz(fmax) if isinstance(fmax, str) else fmax
    f0, _, _ = librosa.pyin(y, fmin=fmin_hz, fmax=fmax_hz, sr=sr)
    times = librosa.times_like(f0, sr=sr)
    voiced_f0 = f0[~np.isnan(f0)]
    if len(voiced_f0) == 0:
        raise ValueError(f"No voiced segments in {audio_path}")
    return {
        'mean_pitch': np.mean(voiced_f0),
        'std_pitch': np.std(voiced_f0),
        'pitch_range': np.ptp(voiced_f0),
        'cv': np.std(voiced_f0) / np.mean(voiced_f0) if np.mean(voiced_f0) != 0 else np.nan,
        'f0': f0, 'times': times
    }

def process_directory(root_folder, sr=16000, fmin='C2', fmax='C7'):
    all_metrics = {}
    plt.figure(figsize=(12, 6))
    for speaker in sorted(os.listdir(root_folder)):
        speaker_path = os.path.join(root_folder, speaker)
        if not os.path.isdir(speaker_path):
            continue
        for audio_file in sorted(glob.glob(os.path.join(speaker_path, "*.wav"))):
            try:
                metrics = compute_pitch_variability(audio_file, sr=sr, fmin=fmin, fmax=fmax)
                all_metrics[audio_file] = metrics
                voiced_mask = ~np.isnan(metrics['f0'])
                if np.any(voiced_mask):
                    label = f"{speaker} - {os.path.basename(audio_file)}"
                    plt.plot(metrics['times'][voiced_mask], metrics['f0'][voiced_mask],
                             marker='o', linestyle='-', markersize=3, label=label)
                print(f"Metrics for {audio_file}: Mean={metrics['mean_pitch']:.2f}, "
                      f"Std={metrics['std_pitch']:.2f}, Range={metrics['pitch_range']:.2f}, "
                      f"CV={metrics['cv']:.2f}")
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (Hz)')
    plt.title('Superimposed Pitch Contours for Processed Audio Clips')
    plt.legend(fontsize='small', loc='upper right', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pitch")
    parser.add_argument('--root_folder', type=str, default='./processed_audio', help="Root folder with speaker subdirectories.")
    parser.add_argument('--sr', type=int, default=16000, help="Sampling rate.")
    parser.add_argument('--fmin', type=str, default='C2', help="Minimum pitch for estimation.")
    parser.add_argument('--fmax', type=str, default='C7', help="Maximum pitch for estimation.")
    args = parser.parse_args()
    
    process_directory(args.root_folder, sr=args.sr, fmin=args.fmin, fmax=args.fmax)
