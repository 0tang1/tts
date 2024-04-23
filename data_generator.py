import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--in_fpath", type=Path, required=True,
                        help="Path to the input audio file used for generating the voice embedding")
    parser.add_argument("--script_fpath", type=Path, required=True,
                        help="Path to the podcast script file")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", default=True, action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)

    ## speech generation from podcast_script.txt

    # reference audio filepath
    #in_fpath = Path('0011_neutral_000002.wav')
    speaker_id = str(args.in_fpath).split('_')[0]
    preprocessed_wav = encoder.preprocess_wav(args.in_fpath)
    print("Loaded files succesfully")
    embed = encoder.embed_utterance(preprocessed_wav)
    print("Created the embeddings")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        synthesizer = Synthesizer(args.syn_model_fpath)

    # Load the script and preprocess the lines
    with open(args.script_fpath, "r") as f:
        for index, line in enumerate(f,start=1):
            if line.strip():
                parts = line.strip().split('\t')
                utterance_id, text, emotion = parts
                texts = [text]
                embeds = [embed]
                specs = synthesizer.synthesize_spectrograms(texts, embeds)
                spec = specs[0]
                print(f"Created the mel spectrogram for {utterance_id}")

                if args.seed is not None:
                    torch.manual_seed(args.seed)
                    vocoder.load_model(args.voc_model_fpath)
                generated_wav = vocoder.infer_waveform(spec)
                generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
                generated_wav = encoder.preprocess_wav(generated_wav)

                if not args.no_sound:
                    import sounddevice as sd
                    try:
                        sd.stop()
                        sd.play(generated_wav, synthesizer.sample_rate)
                    except sd.PortAudioError as e:
                        print("\nCaught exception: %s" % repr(e))
                        print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
                    except:
                        raise

                filename = f"speaker_{speaker_id}/"f"{speaker_id}_neutral_{1750+index:06d}.wav"
                sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
                print("\nSaved output as %s\n\n" % filename)


    
    
    
    
   
