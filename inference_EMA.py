import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
import IPython.display as ipd
import glob
import torch
from pydub import AudioSegment
from torch.utils.data import DataLoader
from text import text_to_sequence, cmudict
from text.symbols import symbols
import utils_data
import re
from num2words import num2words
from kaldiio import WriteHelper
import os
from tqdm import tqdm
from text import text_to_sequence, convert_text
from model import GradTTSWithEmo
import utils_data as utils
from attrdict import AttrDict
from models import Generator as HiFiGAN
from text import _clean_text, symbols


HIFIGAN_CONFIG = './configs/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'


if __name__ == '__main__':
    hps, args = utils.get_hparams_decode()
    device = torch.device('cpu' if not torch.cuda.is_available() else "cuda")
    ckpt = utils_data.latest_checkpoint_path(hps.model_dir, "EMA_grad_*.pt")
    print(ckpt)
    model = GradTTSWithEmo(**hps.model).to(device)
    logger = utils_data.get_logger(hps.model_dir, "inference.log")
    utils_data.load_checkpoint(ckpt, model, None)
    _ = model.cuda().eval()

    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    emos = ["angry", "happy", "neutral", "sad", "suprise"]
    speakers = ['0012', '0011']

    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    replace_nums = []
    cleaned_text = []

    for i in texts:
        replace_nums.append(i.split('|', 1))
    for t in replace_nums:
        text = _clean_text(t[0], cleaner_names=["english_cleaners"]).replace("'", "")
        #print(text)
        cleaned_text.append(text)
    



    nums2word = [re.sub('(\d+)', lambda m: num2words(m.group(), lang='en'), sentence) for sentence in cleaned_text]
   
    text2speech = []
    for i, j in zip(nums2word, np.array(replace_nums)[:, 1]):
        text2speech.append(f'{i}|{j}')

    combined_audio = []

    for i, line in enumerate(text2speech):
        emo_i = int(line.split('|')[1])
        control_spk_id = int(line.split('|')[2])
        control_emo_id = emos.index(emos[emo_i])
        text = line.split('|')[0]
        with torch.no_grad():
            ### define emotion
            emo = torch.LongTensor([control_emo_id]).to(device)
            sid = torch.LongTensor([control_spk_id]).to(device)
            text_padded, text_len = convert_text(text)
            y_enc, y_dec, attn = model.forward(text_padded, text_len, 
                                        n_timesteps=args.timesteps, 
                                        temperature=args.noise,
                                        stoc=args.stoc, spk=sid,emo=emo, length_scale=1.,
                                        classifier_free_guidance=args.guidance)
        res = y_dec.squeeze().cpu().numpy()
        x = torch.from_numpy(res).cuda().unsqueeze(0)
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze()
        audio = audio * 32768.0
        audio_np = audio.detach().cpu().numpy().astype('int16')

   
        padding_duration_seconds = 1  # One second of silence
        samples_to_pad = 22050 * padding_duration_seconds


        padding = np.zeros(samples_to_pad, dtype='int16')

        padded_audio_np = np.concatenate((padding, audio_np))
        audio_segment = AudioSegment(padded_audio_np.tobytes(), frame_rate=16000, sample_width=2, channels=1)
        combined_audio.append(audio_segment)
    # Concatenate all audio segments into one
    final_audio_segment = sum(combined_audio)
    final_audio_segment.export(f'{args.generated_path}/combined_output.wav', format='wav')
