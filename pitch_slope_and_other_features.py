import os
import glob
import argparse
import json
import librosa
import numpy as np
import pandas as pd
import tqdm

def compute_pitch_variability(audio_path, sr=16000, fmin='C2', fmax='C7'):
    y, sr = librosa.load(audio_path, sr=sr)
    fmin_hz = librosa.note_to_hz(fmin) if isinstance(fmin, str) else fmin
    fmax_hz = librosa.note_to_hz(fmax) if isinstance(fmax, str) else fmax
    f0, _, _ = librosa.pyin(y, fmin=fmin_hz, fmax=fmax_hz, sr=sr)
    times = librosa.times_like(f0, sr=sr)
    voiced = ~np.isnan(f0)
    vf0 = f0[voiced]
    if len(vf0) == 0:
        raise ValueError(f"No voiced segments in {audio_path}")
    mean_p, std_p = vf0.mean(), vf0.std()
    prange = np.ptp(vf0)
    cv = std_p / mean_p if mean_p else np.nan
    #vt = times[voiced]
    slopes = np.diff(f0) / np.diff(times) if len(vf0) > 1 else np.array([])
    return {
        'mean_pitch': mean_p,
        'std_pitch': std_p,
        'pitch_range': prange,
        'cv': cv,
        'f0': f0, 
        'times': times, 
        'slopes': slopes
    }

def process_directory(root_folder, sr=16000, fmin='C2', fmax='C7'):
    rows = []
    for wav in tqdm.tqdm(sorted(glob.glob(os.path.join(root_folder, '**', '*.wav'), recursive=True)), desc=root_folder):
        try:
            m = compute_pitch_variability(wav, sr, fmin, fmax)
            rel = os.path.relpath(wav, root_folder).split(os.sep)
            # Expecting path format: dataset/wav/speaker_id/video_id/audio.wav
            if len(rel) >= 4:
                sp, vid = rel[1], rel[2] 
            else:
                sp, vid = '', ''
            row = {
                'speaker': sp,
                'video': vid,
                'file': os.path.basename(wav),
                'mean_pitch': m['mean_pitch'],
                'std_pitch': m['std_pitch'],
                'pitch_range': m['pitch_range'],
                'cv': m['cv'],
                'mean_slope': np.nan if len(m['slopes']) < 1 else m['slopes'].mean(),
                'std_slope': np.nan if len(m['slopes']) < 1 else m['slopes'].std(),
                'slope_range': np.nan if len(m['slopes']) < 1 else np.ptp(m['slopes']),
                'cv_slope': np.nan if len(m['slopes']) < 1 else (
                    m['slopes'].std() / m['slopes'].mean() if m['slopes'].mean() else np.nan
                ),
                'f0': json.dumps(m['f0'].tolist()),
                'times': json.dumps(m['times'].tolist()),
                'slopes': json.dumps(m['slopes'].tolist())
            }
            rows.append(row)
        except Exception as e:
            print(f"Error {wav}: {e}")
    df = pd.DataFrame(rows)
    out = os.path.join(root_folder, 'pitch_features.csv')
    df.to_csv(out, index=False)
    print(f"Saved CSV with arrays serialized to {out}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root_folder', type=str, default='./test_feature_extraction')
    p.add_argument('--sr', type=int, default=16000)
    p.add_argument('--fmin', type=str, default='C2')
    p.add_argument('--fmax', type=str, default='C7')
    args = p.parse_args()
    process_directory(args.root_folder, args.sr, args.fmin, args.fmax)
