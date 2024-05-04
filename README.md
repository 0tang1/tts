This repository is a fork from IS2AI/KazEmoTTS. 

Our ultimate goal is to generate emotionally expressive podcast audio. Below, you'll find step-by-step instructions to replicate our code.

Our example Podcast output can be found at output_full.wav

## Training Data Preparation

Ensure that your data is organized according to the following structure before starting the training process:

### Directory Structure

```plaintext
data/
│
├── [speaker_id] (e.g., 0011, 0012, 0013)
│   ├── [speaker_id].txt - This file contains `[speaker_id]_[utterance_id]` and the corresponding sentence.
│   ├── train/
│   │   └── [speaker_id][emotion][utterance_id].wav - Training audio files.
│   └── val/
│       └── [speaker_id][emotion]_[utterance_id].wav - Validation audio files.
```
### Data Preprocessing

Setup the environment

`conda create -n <env name> python=3.9`

`conda activate <env name>`

Download the nescessary packages

`conda install emotion_env.yml`

`pip install pip_packages.txt`

Run the following command

`python data_preparation.py -d <Your Dataset>`

## Training the GradTTS model

`python train_EMA.py -c configs/train_grad.json -m checkpoint`

## Script Generation

Here make sure to [set up the OpenAI API key](https://openai.com/blog/openai-api) and store it as the `OPENAI_API_KEY` varaible in the bash profile, before running the following code.

`python script_gen.py --topic <your topic>`

The script will be stored as `emotion.txt`

## Podcast Generation

`python inference_EMA.py -c configs/train_grad.json -m checkpoint -t <Number of epochs>  -f <script directory> -r <output directory>`

