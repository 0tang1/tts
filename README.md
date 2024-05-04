This repository is a fork from IS2AI/KazEmoTTS. 

Our ultimate goal is to generate emotionally expressive podcast audio. Below, you'll find step-by-step instructions to replicate our code.

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
## Script Generation

## Podcast Generation

