import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.wav2vec2.processing_wav2vec2")

import argparse
import os
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def main():
    parser = argparse.ArgumentParser(description="Transcribe speech from a WAV file using Wav2Vec2.0.")
    parser.add_argument(
        "wav_file",
        type=str,
        nargs='?',
        default="default.wav",
        help="Path to the WAV file to be transcribed (default: 'default.wav')"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/wav2vec2-large-xlsr-53",
        help="Name or path of the pretrained model to use (default: 'facebook/wav2vec2-large-xlsr-53')"
    )
    
    # Use parse_known_args to ignore unrecognized arguments
    args, unknown = parser.parse_known_args()

    try:
        processor = Wav2Vec2Processor.from_pretrained(args.model_name_or_path)
        model = Wav2Vec2ForCTC.from_pretrained(args.model_name_or_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        audio_input, sample_rate = sf.read(args.wav_file)
    except Exception as e:
        print(f"Error reading the audio file: {e}")
        return

    if sample_rate != 16000:
        print(f"Expected sampling rate 16000, but got {sample_rate}")
        return

    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    try:
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        print("Transcription:", transcription)
    except Exception as e:
        print(f"Error during inference: {e}")

    with open("transcription.txt", "w") as f:
        f.write(transcription)

if __name__ == "__main__":
    main()
