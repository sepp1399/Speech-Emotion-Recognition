import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import random
from tqdm import tqdm
import random

# 📌 Paths
INPUT_DIR = "./DEMozionOVO/audio"
OUTPUT_DIR = "./DEMozionOVO_augmented/audio"
METADATA_FILE = "./DEMozionOVO/metadata/evaluations.csv"
OUTPUT_METADATA_FILE = "./DEMozionOVO_augmented/metadata/evaluations.csv"

# Create output directories if they don’t exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_METADATA_FILE), exist_ok=True)

# Augmentation Functions
def time_stretch(audio, rate=1.0):
    """Applies time stretching to audio."""
    return librosa.effects.time_stretch(y=audio, rate=rate)


def add_gaussian_noise(audio, noise_level=0.005):
    """Adds Gaussian noise."""
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

def apply_vtlp(audio, sr, alpha=1.0):
    """Applies Vocal Tract Length Perturbation (VTLP) by pitch shifting."""
    n_steps = (alpha - 1) * 12  # Convert alpha factor into semitone shift
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def apply_rir(audio):
    """Simulates Room Impulse Response (RIR) by applying convolution with a small echo."""
    rir = np.random.randn(int(len(audio) * 0.1)) * 0.02  # Simulated impulse response
    return np.convolve(audio, rir, mode='same')

# Load metadata
df_metadata = pd.read_csv(METADATA_FILE)

# shuffle
df_metadata = df_metadata.sample(frac=1, random_state=None).reset_index(drop=True)

# Augmentation processing
augmented_rows = []
emotion_counts = {emotion: 0 for emotion in df_metadata["emotion_expressed"].unique()}
MAX_AUG_PER_EMOTION = 100  # Maximum per technique per emotion

for _, row in tqdm(df_metadata.iterrows(), desc="Processing audio files", total=len(df_metadata)):
    file_name, actor, emotion = row["file_name"], row["actor"], row["emotion_expressed"]
    file_path = os.path.join(INPUT_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    y, sr = librosa.load(file_path, sr=None)
    
    # Apply augmentations
    if emotion_counts[emotion] < MAX_AUG_PER_EMOTION:
        stretched = time_stretch(y, rate=random.uniform(0.8, 1.2))
        noisy = add_gaussian_noise(y, noise_level=random.uniform(0.002, 0.01))
        
        stretched_name = f"stretched_{file_name}"
        noisy_name = f"noisy_{file_name}"
        
        sf.write(os.path.join(OUTPUT_DIR, stretched_name), stretched, sr)
        sf.write(os.path.join(OUTPUT_DIR, noisy_name), noisy, sr)
        
        augmented_rows.append([stretched_name, actor, emotion])
        augmented_rows.append([noisy_name, actor, emotion])
        
        emotion_counts[emotion] += 1
    
    if emotion_counts[emotion] < MAX_AUG_PER_EMOTION:
        rir_applied = apply_rir(y)

        alpha = random.choice([random.uniform(0.75, 0.8), random.uniform(1.2, 1.25)])
        vtlp_applied = apply_vtlp(y, sr, alpha=alpha)
        
        rir_name = f"rir_{file_name}"
        vtlp_name = f"vtlp_{file_name}"
        
        sf.write(os.path.join(OUTPUT_DIR, rir_name), rir_applied, sr)
        sf.write(os.path.join(OUTPUT_DIR, vtlp_name), vtlp_applied, sr)
        
        augmented_rows.append([rir_name, actor, emotion])
        augmented_rows.append([vtlp_name, f"{actor}_VTLP", emotion])  # Modify actor for VTLP
        
        emotion_counts[emotion] += 1

# Save updated metadata
df_augmented = pd.DataFrame(augmented_rows, columns=["file_name", "actor", "emotion_expressed"])
df_final = pd.concat([df_metadata, df_augmented], ignore_index=True)
df_final.to_csv(OUTPUT_METADATA_FILE, index=False)

print("Augmentation and metadata update completed!")
