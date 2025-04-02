import argparse
import os

import numpy as np
import torch
from torch.nn import functional as F

import utils
from data_loader import MelSpectrogramFixed
from inference import get_yaapt_f0, load_audio
from model.vc_dddm_mixup import DDDM, Wav2vec2
from model_f0_vqvae import Quantizer
from vocoder.hifigan import HiFi

mel_fn, w2v, f0_quantizer, model, net_v = None, None, None, None, None


def get_speaker_embedding(a):
    global mel_fn, w2v, f0_quantizer, model, net_v

    config = os.path.join(os.path.split(a.ckpt_model)[0], "config.json")
    hps = utils.get_hparams_from_file(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mel_fn is None:
        mel_fn = MelSpectrogramFixed(
            sample_rate=hps.data.sampling_rate,
            n_fft=hps.data.filter_length,
            win_length=hps.data.win_length,
            hop_length=hps.data.hop_length,
            f_min=hps.data.mel_fmin,
            f_max=hps.data.mel_fmax,
            n_mels=hps.data.n_mel_channels,
            window_fn=torch.hann_window,
        ).cuda()

    if w2v is None:
        # Load pre-trained w2v (XLS-R)
        w2v = Wav2vec2().cuda()

    if f0_quantizer is None:
        # Load model
        f0_quantizer = Quantizer(hps).cuda()
        utils.load_checkpoint(a.ckpt_f0_vqvae, f0_quantizer)
        f0_quantizer.eval()

    if model is None:
        model = DDDM(
            hps.data.n_mel_channels,
            hps.diffusion.spk_dim,
            hps.diffusion.dec_dim,
            hps.diffusion.beta_min,
            hps.diffusion.beta_max,
            hps,
        ).cuda()
        utils.load_checkpoint(a.ckpt_model, model, None)
        model.eval()

    if net_v is None:
        # Load vocoder
        net_v = HiFi(
            hps.data.n_mel_channels,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
        ).cuda()
        utils.load_checkpoint(a.ckpt_voc, net_v, None)
        net_v.eval().dec.remove_weight_norm()

    # Convert audio
    # print(">> Converting each utterance...")
    src_name = os.path.splitext(os.path.basename(a.src_path))[0]
    audio = load_audio(a.src_path)

    src_mel = mel_fn(audio.cuda())
    src_length = torch.LongTensor([src_mel.size(-1)]).cuda()
    w2v_x = w2v(F.pad(audio, (40, 40), "reflect").cuda())

    try:
        f0 = get_yaapt_f0(audio.numpy())
    except:
        f0 = np.zeros((1, audio.shape[-1] // 80), dtype=np.float32)

    ii = f0 != 0
    f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()
    f0 = torch.FloatTensor(f0).cuda()
    f0_code = f0_quantizer.code_extraction(f0)

    trg_name = os.path.splitext(os.path.basename(a.trg_path))[0]
    trg_audio = load_audio(a.trg_path)

    trg_mel = mel_fn(trg_audio.cuda())
    trg_length = torch.LongTensor([trg_mel.size(-1)]).to(device)

    with torch.no_grad():
        c = model.encode_speaker(
            w2v_x,
            f0_code,
            src_length,
            trg_mel,
            trg_length,
        )
        return c.cpu().detach().numpy()
