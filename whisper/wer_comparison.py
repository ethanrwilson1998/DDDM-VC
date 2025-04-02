import os
import jiwer
from glob import glob
import numpy as np

def load_transcript(file_path):
    """Loads text from a transcript file."""
    if not os.path.exists(file_path):
        return None  # Return None if the file does not exist
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def calculate_wer(original_folder, anonymized_folder):
    """
    Calculate WER for each speaker in the anonymized folder against the original.

    Parameters:
    - original_folder: Path to the original transcripts.
    - anonymized_folder: Path to the anonymized transcript folder.
    """
    speakers = sorted(os.listdir(original_folder))  # Get list of speaker IDs
    wer_results = {}

    for speaker in speakers:
        original_speaker_path = os.path.join(original_folder, speaker)
        anonymized_speaker_path = os.path.join(anonymized_folder, speaker)

        if not os.path.isdir(original_speaker_path) or not os.path.isdir(anonymized_speaker_path):
            continue  # Skip if the directory is missing in either original or anonymized set

        # Get all transcript files in the original folder
        original_transcripts = sorted(glob(os.path.join(original_speaker_path, "*.txt")))

        for original_transcript in original_transcripts:
            filename = os.path.basename(original_transcript)  # Get filename (e.g., "file1.txt")
            original_text = load_transcript(original_transcript)  # Load original transcript

            if original_text is None:
                continue  # Skip if original transcript is missing

            anonymized_transcript = os.path.join(anonymized_speaker_path, filename)
            anonymized_text = load_transcript(anonymized_transcript)  # Load anonymized transcript

            if anonymized_text is None:
                continue  # Skip if anonymized transcript is missing

            # Compute WER
            wer = jiwer.wer(original_text, anonymized_text)

            # Store WER for this speaker
            if speaker not in wer_results:
                wer_results[speaker] = []

            wer_results[speaker].append(wer)

    # Print results
    print(f"\n>> WER Comparison: {original_folder} vs. {anonymized_folder}")
    for speaker, wer_scores in wer_results.items():
        avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0
        print(f"  Speaker {speaker}: {avg_wer:.2%}")
        net_wer = np.average(avg_wer)
        print(net_wer)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--original_folder', default = './voxceleb_epsilon_1', help="Path to the original transcript directory")
    parser.add_argument('--anonymized_folder', default = './voxceleb_epsilon_1', help="Path to the anonymized transcript directory")

    args = parser.parse_args()
    
    calculate_wer(args.original_folder, args.anonymized_folder)
