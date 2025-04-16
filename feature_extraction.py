import os
import glob
import argparse
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
    voiced_mask = ~np.isnan(f0)
    voiced_f0 = f0[voiced_mask]
    if len(voiced_f0) == 0:
        raise ValueError(f"No voiced segments in {audio_path}")
    mean_pitch = np.mean(voiced_f0)
    std_pitch = np.std(voiced_f0)
    pitch_range = np.ptp(voiced_f0)
    cv = std_pitch / mean_pitch if mean_pitch != 0 else np.nan
    valid_times = times[voiced_mask]
    if len(voiced_f0) < 2:
        slopes = np.array([])
        mean_slope = np.nan
        std_slope = np.nan
        slope_range = np.nan
        cv_slope = np.nan
    else:
        slopes = np.diff(voiced_f0) / np.diff(valid_times)
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        slope_range = np.ptp(slopes)
        cv_slope = std_slope / mean_slope if mean_slope != 0 else np.nan
    return {
        'mean_pitch': mean_pitch,
        'std_pitch': std_pitch,
        'pitch_range': pitch_range,
        'cv': cv,
        'f0': f0,
        'times': times,
        'mean_slope': mean_slope,
        'std_slope': std_slope,
        'slope_range': slope_range,
        'cv_slope': cv_slope
    }

def process_directory(root_folder, sr=16000, fmin='C2', fmax='C7'):
    all_rows = []
    file_list = glob.glob(os.path.join(root_folder, '**', '*.wav'), recursive=True)
    for audio_file in tqdm.tqdm(sorted(file_list), desc=root_folder):
        try:
            metrics = compute_pitch_variability(audio_file, sr=sr, fmin=fmin, fmax=fmax)
            rel_path = os.path.relpath(audio_file, root_folder)
            parts = rel_path.split(os.sep)
            speaker = parts[0]
            video = parts[1] if len(parts) >= 3 else ""
            file_name = os.path.basename(audio_file)
            row = {
                "speaker": speaker,
                "video": video,
                "file": file_name,
                "mean_pitch": metrics['mean_pitch'],
                "std_pitch": metrics['std_pitch'],
                "pitch_range": metrics['pitch_range'],
                "cv": metrics['cv'],
                "mean_slope": metrics['mean_slope'],
                "std_slope": metrics['std_slope'],
                "slope_range": metrics['slope_range'],
                "cv_slope": metrics['cv_slope']
            }
            all_rows.append(row)
            # print(f"{audio_file}: Mean={metrics['mean_pitch']:.2f}, Std={metrics['std_pitch']:.2f}, "
            #       f"Range={metrics['pitch_range']:.2f}, CV={metrics['cv']:.2f}, "
            #       f"MeanSlope={metrics['mean_slope']:.2f}, StdSlope={metrics['std_slope']:.2f}, "
            #       f"SlopeRange={metrics['slope_range']:.2f}, CVSlope={metrics['cv_slope']:.2f}")
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    df = pd.DataFrame(all_rows)
    output_csv = os.path.join(root_folder, "pitch_features.csv")
    df.to_csv(output_csv, index=False)
    print(f"Features spreadsheet saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pitch Feature Extraction")
    parser.add_argument('--root_folder', type=str, default='./test_feature_extraction',
                        help="Root folder with speaker subdirectories.")
    parser.add_argument('--sr', type=int, default=16000, help="Sampling rate.")
    parser.add_argument('--fmin', type=str, default='C2', help="Minimum pitch for estimation.")
    parser.add_argument('--fmax', type=str, default='C7', help="Maximum pitch for estimation.")
    args = parser.parse_args()
    process_directory(args.root_folder, sr=args.sr, fmin=args.fmin, fmax=args.fmax)
