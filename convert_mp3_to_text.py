import os
import sys
import subprocess
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

# Check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Whisper model and processor from Hugging Face
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Set up the pipeline for automatic speech recognition
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,  # Process in 30-second chunks
    batch_size=16,  # Adjust based on your GPU or CPU capacity
    torch_dtype=torch_dtype,
    device=device,
)

def convert_mp3_to_wav(mp3_file, wav_file):
    """Convert MP3 file to WAV with 16kHz mono using ffmpeg"""
    subprocess.run(["ffmpeg", "-i", mp3_file, "-ar", "16000", "-ac", "1", wav_file], check=True)

def transcribe_wav(wav_file):
    """Transcribe the WAV file using Whisper model"""
    # Load the audio file
    audio_input, sample_rate = sf.read(wav_file)
    
    # Ensure that the sample rate is 16kHz as expected
    assert sample_rate == 16000, "Audio file must have a 16kHz sample rate."

    # Perform the transcription in chunks and return the full transcription
    result = pipe(audio_input, return_timestamps=False)
    return result["text"]

def main(mp3_file):
    # Ensure the MP3 file exists
    if not os.path.exists(mp3_file):
        print(f"Error: {mp3_file} does not exist!")
        sys.exit(1)
    
    # Set the output file paths
    base_name = os.path.splitext(mp3_file)[0]
    wav_file = base_name + ".wav"
    transcription_file = base_name + ".txt"
    
    # Convert MP3 to WAV
    print(f"Converting {mp3_file} to WAV...")
    convert_mp3_to_wav(mp3_file, wav_file)
    
    # Transcribe the WAV file
    print(f"Transcribing {wav_file}...")
    transcription = transcribe_wav(wav_file)
    
    # Save the transcription to a text file
    with open(transcription_file, "w") as f:
        f.write(transcription)
    
    print(f"Transcription saved to {transcription_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_mp3_to_text.py <mp3_file>")
        sys.exit(1)
    
    mp3_file = sys.argv[1]
    main(mp3_file)