import os
import whisper
import glob
import tqdm

model = None


def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    transcript_path = audio_path.replace(".wav", ".txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    # print(f"Transcript saved: {transcript_path}")

def process_folder(a):
    global model
    model = whisper.load_model(a.model)

    wav_files = glob.glob(f"{a.audio_folder}/**/*.wav", recursive=True)
        
    for clip in tqdm.tqdm(wav_files):
        # print(f'Transcribing {clip}...')
        transcribe_audio(clip)  
    
    print(">> Transcription Process Complete.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_folder', type=str, default = './voxceleb_base', help='Path to folder containing participant audio')
    parser.add_argument('--model', type=str, choices=['tiny.en', 'small.en', 'medium.en', 'turbo'], default='turbo', help='name of whisper model to use.')
    args = parser.parse_args()
    
    process_folder(args)
