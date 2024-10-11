#!/bin/bash

# Folder containing MP3 files
mp3_folder="voice_recordings"

# Loop through all MP3 files in the folder
for mp3_file in "$mp3_folder"/*.mp3; do
    echo "Processing $mp3_file ..."
    # Call the Python script for each MP3 file
    python3 convert_mp3_to_text.py "$mp3_file"
done

echo "All MP3 files have been processed!"
