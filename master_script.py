"""
TITLE: "Master Script for Speech Emotion Recognition with Wav2Vec2"
DESCRIPTION: This script orchestrates the execution of multiple Python scripts for training and testing a Wav2Vec2 model on speech emotion recognition tasks.
AUTHORS: Giuseppe Lentini
LAST UPDATE: 2025-06-23
PYTHON VERSION: 3.11.11
"""
import subprocess

scripts = [
    '01_augmentation_finder.py',
    '02_transformer_training.py',
    '03_transformer_testing.py',
    '04_model_converter.py'
]

for script in scripts:
    print(f"\n🔹 Running: {script}")
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {script}: exit code {e.returncode}")