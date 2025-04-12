import os
import jiwer
import glob
import numpy as np
import tqdm

def load_transcript(file_path):
    """Loads text from a transcript file."""
    if not os.path.exists(file_path):
        return None  # Return None if the file does not exist
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def calculate_wer(a):
    """
    Calculate WER for each speaker in the anonymized folder against the original.

    Parameters:
    - original_folder: Path to the original transcripts.
    - anonymized_folder: Path to the anonymized transcript folder.
    """
    # NOTE: I removed speaker info since we're now doing full test set

    o_sents, a_sents = [], []

    original_transcripts = glob.glob(f"{a.original_folder}/**/*.txt", recursive=True)

    for ot in tqdm.tqdm(original_transcripts):
        at = ot.replace(a.original_folder, a.anonymized_folder)

        original_text = load_transcript(ot)
        anonymized_text = load_transcript(at)
        if original_text is None:
            original_text = ""
        if anonymized_text is None:
            anonymized_text = ""

        o_sents.append(original_text)
        a_sents.append(anonymized_text)

    # Compute WER
    wer = jiwer.wer(o_sents, a_sents)

    # Print results
    print(f"\n>> WER Comparison: {a.original_folder} vs. {a.anonymized_folder}")
    print(f"\tWER: {wer:.2%}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--original_folder', default = './voxceleb_epsilon_1', help="Path to the original transcript directory")
    parser.add_argument('--anonymized_folder', default = './voxceleb_epsilon_1', help="Path to the anonymized transcript directory")

    args = parser.parse_args()
    
    calculate_wer(args)
