import argparse
import glob
from multiprocessing import process
import ntpath
import os

import tqdm

from inference import inference


def process_folder(a, folder):

    scp_file = f"{a.data_dir}/data/{folder}/wav.scp"

    with open(scp_file, 'r') as file:
        for line in file:
            to_write, ext_file = line.split(" ")

            if a.method == "VoiceVMF":
                to_file = f"{a.data_dir}/{folder}_VoiceVMF_e{a.epsilon}/{to_write}.wav"
            elif a.method == "IdentityDP":
                to_file = f"{a.data_dir}/{folder}_IdentityDP_e{a.epsilon}/{to_write}.wav"

            a.src_path = f"{a.data_dir}{ext_file}"
            a.trg_path = a.src_path
            a.output_path = to_file
            inference(a)

def process_vpc(a):

    process_folder("libri_dev_enrolls")
    process_folder("libri_dev_trials_m")
    process_folder("libri_dev_trials_f")
    process_folder("libri_test_enrolls")
    process_folder("libri_test_trials_m")
    process_folder("libri_test_trials_f")
    process_folder("IEMOCAP_dev")
    process_folder("IEMOCAP_test")
    process_folder("train-clean-360")

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

    process_folder(a)


if __name__ == "__main__":
    main()
