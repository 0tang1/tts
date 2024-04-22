import kaldiio
import os
import librosa
from tqdm import tqdm
import glob
import json 
from shutil import copyfile
import pandas as pd
import argparse
from text import _clean_text, symbols
from num2words import num2words
import re
from melspec import mel_spectrogram
import torchaudio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help='path to the emotional dataset')
    args = parser.parse_args()
    dataset_path = args.data
    filelists_path = 'filelists/all_spks/'
    feats_scp_file = filelists_path + 'feats.scp'
    feats_ark_file = filelists_path + 'feats.ark'


    # 写一个loop存 speaker id
    spks = []
    for spk in os.listdir(dataset_path):
        spks.append(spk)
    print(f"spks:{spks}")
    
    train_files = []
    eval_files = []

    wavs = [] #storing all the paths for wav files

    # 存train和test的directory
    for spk in spks:
        
        train_wavs = glob.glob(os.path.join(dataset_path, spk, "train", "*.wav"))
        eval_wavs = glob.glob(os.path.join(dataset_path, spk, "eval", "*.wav")) 

        train_files.extend(train_wavs)
        eval_files.extend(eval_wavs)

        wavs.extend(train_wavs)
        wavs.extend(eval_wavs)

   
    os.makedirs(filelists_path, exist_ok=True)

    # Create txt file storing the file names of wav，因为原始wav name “speakerid_emotion_utterance”
    train_utts_path = os.path.join(filelists_path, 'train_utts.txt')
    with open(train_utts_path, 'w', encoding='utf-8') as f:
        for wav_path in train_files:
            wav_name = os.path.splitext(os.path.basename(wav_path))[0]
            
            f.write(wav_name + '\n')
    eval_utts_path = os.path.join(filelists_path, 'eval_utts.txt')
    with open(eval_utts_path, 'w', encoding='utf-8') as f:
        for wav_path in eval_files:
            wav_name = os.path.splitext(os.path.basename(wav_path))[0]
            
            f.write(wav_name + '\n')

    # 转成spectrogram
    with open(feats_scp_file, 'w') as feats_scp, \
        kaldiio.WriteHelper(f'ark,scp:{feats_ark_file},{feats_scp_file}') as writer:
        for root, dirs, files in os.walk(dataset_path):
            for file in tqdm(files):
                if file.endswith('.wav'):
                    # Get the file name and relative path to the root folder
                    wav_path = os.path.join(root, file)
                    rel_path = os.path.relpath(wav_path, dataset_path)
                    wav_name = os.path.splitext(os.path.basename(wav_path))[0]
                    signal, rate = torchaudio.load(wav_path)
                    spec = mel_spectrogram(signal, 1024, 80, 22050, 256,
                              1024, 0, 8000, center=False).squeeze()
                    # Write the features to feats.ark and feats.scp
                    writer[wav_name] = spec
    

    # emotion is the 2nd element of the file name
    emotions = [os.path.basename(x).split("_")[1] for x in glob.glob(dataset_path + '/**/**/*')]
    emotions = sorted(set(emotions))

    # 根据wave_name “speakerid_emotion_utterance” 作为key，emotion，spk作为values
    utt2spk = {}
    utt2emo = {}
    
    print(f"Found {len(wavs)} .wav files.")

    for wav_path in tqdm(wavs):
        wav_name = os.path.splitext(os.path.basename(wav_path))[0]
        emotion =  emotions.index(wav_name.split("_")[1])

        speaker = wav_name.split("_")[0]
        spk = spks.index(speaker)

        utt2spk[wav_name] = str(spk)
        utt2emo[wav_name] = str(emotion)
    utt2spk = dict(sorted(utt2spk.items()))
    utt2emo = dict(sorted(utt2emo.items()))
    print("Size of utt2spk:", len(utt2spk))

    with open(filelists_path + 'utt2emo.json', 'w') as fp:
        json.dump(utt2emo, fp,  indent=4)
    with open(filelists_path + 'utt2spk.json', 'w') as fp:
        json.dump(utt2spk, fp,  indent=4) 

    #改！
    # text file storing the wav name “speakerid_emotion_utterance” sentence
    txt_files = sorted(glob.glob(dataset_path + '/**/*.txt'))
    combined_lines = []

    for txt_file in sorted(glob.glob(os.path.join(dataset_path, '**/*.txt'), recursive=True)):
        speaker_id = os.path.basename(os.path.dirname(txt_file))  # Extract the speaker ID from the parent directory name
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  

                    parts = line.strip().split("\t")
                    utterance_id, text, emotion = parts
                    # Process the text through the cleaning functions
                    cleaned_text = _clean_text(text, cleaner_names=["english_cleaners"]).replace("'", "")
                    cleaned_text = re.sub('(\d+)', lambda m: num2words(m.group(), lang='en'), cleaned_text)
                    cleaned_text = ''.join([s for s in cleaned_text if s in symbols])

                    # Extract the utterance number and rebuild the utterance ID
                    utterance_number = utterance_id.split('_')[-1]
                    new_utterance_id = f"{speaker_id}_{emotion.lower()}_{utterance_number}"
                    combined_line = f"{new_utterance_id}\t{cleaned_text}\n"
                    combined_lines.append(combined_line)

    with open(filelists_path + '/text', 'w', encoding='utf-8') as f:
        for line in combined_lines:
            f.write(line)

