import argparse
import glob
import os

import tqdm

from inference import inference


def process_folder(a):
    audio_clips = glob.glob(f"{a.audio_folder}/**/*.wav", recursive=True)

    for clip in tqdm.tqdm(audio_clips):
        folders = clip.split(os.sep)
        dataset_name = folders[-4]
        if a.method == "VoiceVMF":
            anon_dataset_name = f"{dataset_name}_VoiceVMF_e{a.epsilon}"

        new_clip = clip.replace(dataset_name, anon_dataset_name)

        print(f"Processing {clip}...")
        a.src_path = clip
        if a.method in ["VoiceVMF"]:
            a.trg_path = (
                clip  # Since we are processing per participant, keep trg_path same
            )
        a.output_path = new_clip
        inference(a)

    print(">> Processing Complete.")


def main():
    print(">> Initializing Inference Process...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_folder",
        type=str,
        default="./voxceleb",
        help="Path to folder containing participant audio",
    )
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
