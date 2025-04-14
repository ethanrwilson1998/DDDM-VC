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
    mean_pitch = np.mean(voiced_f0)
    std_pitch = np.std(voiced_f0)
    pitch_range = np.ptp(voiced_f0)
    cv = std_pitch / mean_pitch if mean_pitch != 0 else np.nan
    return {'mean_pitch': mean_pitch, 'std_pitch': std_pitch,
            'pitch_range': pitch_range, 'cv': cv, 'f0': f0, 'times': times}

if __name__ == '__main__':
    audio_files = [
        'sample/00001_og.wav', 
        'sample/00001_e1.0.wav',
        'sample/00001_e10.wav',
        'sample/00001_e50.wav'
    ]
    all_metrics = {}
    plt.figure(figsize=(10, 4))
    for audio_file in audio_files:
        try:
            metrics = compute_pitch_variability(audio_file)
            all_metrics[audio_file] = metrics
            voiced_mask = ~np.isnan(metrics['f0'])
            if np.any(voiced_mask):
                plt.plot(metrics['times'][voiced_mask], metrics['f0'][voiced_mask],
                         marker='o', linestyle='-', markersize=3, label=audio_file)
            print(f"Metrics for {audio_file}: Mean={metrics['mean_pitch']:.2f}, Std={metrics['std_pitch']:.2f}, "
                  f"Range={metrics['pitch_range']:.2f}, CV={metrics['cv']:.2f}")
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (Hz)')
    plt.title('Superimposed Pitch Contours for Different Anonymization Levels')
    plt.legend()
    plt.grid(True)
    plt.show()
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
    mean_pitch = np.mean(voiced_f0)
    std_pitch = np.std(voiced_f0)
    pitch_range = np.ptp(voiced_f0)
    cv = std_pitch / mean_pitch if mean_pitch != 0 else np.nan
    return {'mean_pitch': mean_pitch, 'std_pitch': std_pitch,
            'pitch_range': pitch_range, 'cv': cv, 'f0': f0, 'times': times}

if __name__ == '__main__':
    audio_files = [
        'sample/00001_og.wav', 
        'sample/00001_e1.0.wav',
        'sample/00001_e10.wav',
        'sample/00001_e50.wav'
    ]
    all_metrics = {}
    plt.figure(figsize=(10, 4))
    for audio_file in audio_files:
        try:
            metrics = compute_pitch_variability(audio_file)
            all_metrics[audio_file] = metrics
            voiced_mask = ~np.isnan(metrics['f0'])
            if np.any(voiced_mask):
                plt.plot(metrics['times'][voiced_mask], metrics['f0'][voiced_mask],
                         marker='o', linestyle='-', markersize=3, label=audio_file)
            print(f"Metrics for {audio_file}: Mean={metrics['mean_pitch']:.2f}, Std={metrics['std_pitch']:.2f}, "
                  f"Range={metrics['pitch_range']:.2f}, CV={metrics['cv']:.2f}")
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (Hz)')
    plt.title('Superimposed Pitch Contours for Different Anonymization Levels')
    plt.legend()
    plt.grid(True)
    plt.show()
