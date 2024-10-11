# Voice to Text Transcription Tool

This tool is designed to process a folder containing voice recordings and transcribe them into text using the OpenAI Whisper model leveraging Hugging Face. 

## Features

- Converts MP3 files to WAV format using `ffmpeg`.
- Transcribes the WAV files into text using the Whisper model, leveraging  Hugging Face library. Whisper is classified as an Audio / Automatic Speech Recognition model (ASR).
- Supports the processing of longer audio recordings by chunking them into manageable 30-second segments for transcription.
- The turbo model (`openai/whisper-large-v3-turbo`) is designed to balance speed and accuracy

## Handling Longer Recordings

This tool handles longer recordings (longer than 10 seconds) by chunking the audio into 30-second segments, ensuring that each chunk is processed without overloading the memory. This approach allows for seamless transcription of long recordings while maintaining efficiency.

## Installation and Setup

```bash
git clone ...
python3 -m venv venv
source ./venv/bin/activate
pip install torch soundfile transformers tqdm
pip install 'accelerate>=0.26.0'
./transcribe_all_mp3s.sh
```

## Whisper model

1. Whisper is an open weights model. The trained model weights and inference code is published publicly. The training code is not published publicly.

2. Whisper is a Transformer-based LLM designed for speech recognition, leveraging the same architecture as large language models (LLMs) to process audio as a sequence of tokens, enabling tasks like multilingual speech recognition, translation, and language identification.

3. Trained on 680K hours of multilingual audio data, Whisper offers robust speech recognition via large-scale Weak Supervision. About a third of the training data is non-English, improving its multilingual capabilities.



