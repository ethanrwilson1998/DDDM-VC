import argparse
import glob
import ntpath
import os

import librosa
import soundfile as sf
import tqdm

from inference import inference

import logging
logging.getLogger('numba').setLevel(logging.WARNING)


def process_folder(a, folder):

    scp_file = f"{a.data_dir}/data/{folder}/wav.scp"

    with open(scp_file, 'r') as f:
        print(scp_file)
        for line in tqdm.tqdm(f, desc=folder):
            line_items = line.split(" ")
            if "train-clean-360" in folder:
                to_write, ext_file = line_items[0], line_items[-2]
            else:
                to_write, ext_file = line_items[0], line_items[1]

            if a.method == "VoiceVMF":
                to_file = f"{a.data_dir}/data/{folder}_VoiceVMF_e{a.epsilon}/wav/{to_write}.wav"
            elif a.method == "IdentityDP":
                to_file = f"{a.data_dir}/data/{folder}_IdentityDP_e{a.epsilon}/wav/{to_write}.wav"

            a.src_path = f"{a.data_dir}/{ext_file}"
            a.src_path = a.src_path.replace("\n", "")
            a.trg_path = a.src_path
            a.output_path = to_file

            if not os.path.exists(a.output_path):
                if ".flac" in a.src_path:
                    wav_file, sr = librosa.load(a.src_path, sr=None)
                    a.src_path = f"{a.data_dir}/data/tmp.wav"
                    a.trg_path = a.src_path
                    sf.write(a.src_path, wav_file, sr)

                inference(a)

def process_vpc(a):

    process_folder(a, "libri_dev_enrolls")
    process_folder(a, "libri_dev_trials_m")
    process_folder(a, "libri_dev_trials_f")
    process_folder(a, "libri_test_enrolls")
    process_folder(a, "libri_test_trials_m")
    process_folder(a, "libri_test_trials_f")
    process_folder(a, "IEMOCAP_dev")
    process_folder(a, "IEMOCAP_test")
    # TODO: do this last after all conditions have prior sets generated
    #process_folder(a, "train-clean-360")

    print(">> Processing Complete.")


def main():
    print(">> Initializing Inference Process...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/blue/ejain/ethanwilson/Voice-Privacy-Challenge-2024")
    parser.add_argument("--ckpt_model", type=str, default="./ckpt/model_base.pth")
    parser.add_argument("--ckpt_voc", type=str, default="./vocoder/voc_ckpt.pth")
    parser.add_argument(
        "--ckpt_f0_vqvae", "-f", type=str, default="./f0_vqvae/f0_vqvae.pth"
    )
    parser.add_argument("--time_step", "-t", type=int, default=6)
    parser.add_argument(
        "--method", choices=["swap", "VoiceVMF", "IdentityDP"], type=str, default="swap"
    )
    parser.add_argument("--epsilon", type=float, default=0)
    parser.add_argument("--theta", type=float, default=0)

    a = parser.parse_args()

    process_vpc(a)


if __name__ == "__main__":
    main()
