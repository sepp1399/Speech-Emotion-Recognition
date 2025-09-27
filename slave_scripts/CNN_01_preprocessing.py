"""
TITLE: ""
AUTHOR: Giuseppe Lentini, Paolo Ranzi
LAST UPDATE: 20250219
PYTHON VERSION: 3.10.6

DESCRIPTION: 
Please change the following sections according to your individual input preferences:
    - '2. PARAMETERS TO BE SET!!!'

"""

###############################################################################
## 1. IMPORTING LIBRARIES
import os, glob
import numpy as np
import pandas as pd
import torchaudio
import torch
import time
import gc

###############################################################################
## 2. PARAMETERS TO BE SET!!!

# set the correct pathways/folders
BASE_DIR_INPUT = ('.')
BASE_DIR_OUTPUT = BASE_DIR_INPUT 

model_number = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         'config/model_number.csv']), header = None, 
                                     dtype = 'str').iloc[0].values[0]
# set input/output file names set input/output file names
input_file_name_05 = ('config/run_modality.csv')
output_file_name_28 = ('output/data/')

# set dataset used
DATASET = 'Emozionalmente_augmented_core'

# # delete old output only if we are on 1th iteration
# if os.path.exists('./output') and model_number == 'MODEL_01':
#     shutil.rmtree('./output')

# Create new output directory
os.makedirs('./output', exist_ok=True)

# setup logging
LOG_DIR = os.path.join("output", model_number, "log")
LOG_FILE = os.path.join(LOG_DIR, "01_preprocessing_log.txt")

# Create log directory 
os.makedirs(LOG_DIR, exist_ok=True)

# Create processed_data directory if it doesn't exist
os.makedirs('./processed_data', exist_ok=True)

# log function
def log(message, do_print=True):
    """Writes a message to both the console and the log file."""
    
    if do_print:
        print(message)

    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

###############################################################################
## 3. LOADING DATA-SET 
# start clocking time
start_time = time.time()

# notify pre-processing intialization
log("=== PREPROCESSING START ===")
log(f"Start time (Unix timestamp): {start_time}")
log("Extracting MFCC, Delta, Delta-Delta, ZCR and F0 features from audio files...")
log("=================================\n")

# run_modality, either: 
# - 'testing' : without grid-search => quick; 
# - 'production' : with grid-search => slow;
run_modality = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         input_file_name_05]), header = None, 
                                     dtype = 'str')

# check if "output" folder exists, otherwise creates one
# Indeed, this is a workaround for GitHub tendency to delete an empty folder... 
output_dir = os.path.join(BASE_DIR_OUTPUT, "output", model_number, "data")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

################################################################################
## 4. PRE-PROCESSING

# set sampling rate
SAMPLING_RATE = 16000

# set number of MFCC features
N_MFCC = 30

# set MFCC parameters
MFCC_PARAMS = {
    "n_fft": 1024,
    "n_mels": 80,
    "hop_length": 256,
    "mel_scale": "htk"
}

log("=== PREPROCESSING PARAMETERS ===")
log(f"Dataset used: {DATASET}")
log(f"Sampling rate: {SAMPLING_RATE}")
log(f"MFCC parameters: {MFCC_PARAMS}")
log(f"Number of MFCCs: {N_MFCC}")
log("=================================\n")

def compute_zero_crossing_rate(waveform, frame_length=2048, hop_length=512):
    """
    Compute Zero-Crossing Rate (ZCR) using PyTorch, mimicking librosa.feature.zero_crossing_rate.

    Parameters:
    - waveform: Tensor (1, samples) -> Audio waveform
    - frame_length: Size of each analysis frame (default 2048)
    - hop_length: Step size between frames (default 512)

    Returns:
    - zcr: Tensor (1, time_steps) -> Zero-Crossing Rate per frame
    """
    # Ensure waveform is (1, samples)
    waveform = waveform.squeeze(0)  

    # Compute zero crossings (1 when sign change occurs)
    zero_crossings = torch.diff(torch.sign(waveform)) != 0  
    zero_crossings = zero_crossings.float()

    # Frame the signal using unfold (similar to librosa's framing)
    frames = waveform.unfold(dimension=0, size=frame_length, step=hop_length)  

    # Count zero crossings in each frame
    zcr = torch.mean(torch.abs(torch.diff(torch.sign(frames), dim=-1)), dim=-1)

    zcr = torch.log1p(zcr)  # log(1 + ZCR) to avoid log(0) issues to stabilize distribution and reduce the impact of outliers.

    return zcr.unsqueeze(0)  # Shape: (1, time_steps)

# compute MFCC features
def extract_feature(file_name, sampling_rate):
    """
    Extract MFCC, Delta, and Delta-Delta features from an audio file.

    # Input:
    - file_name: Path to the .wav file
    - sampling_rate: Target sampling rate (e.g., 8000 Hz)

    # Output:
    - result: Feature vector (MFCC + Delta + Delta-Delta)
    """
    
    # Initialize Numpy array for storing features
    result = np.array([])

    # Load and preprocess audio clip
    wav, sr = preprocess_audio(file_name, sampling_rate)

    # Initialize MFCC features
    mfcc_step = torchaudio.transforms.MFCC(
        sample_rate=sampling_rate, 
        n_mfcc=N_MFCC,
        melkwargs=MFCC_PARAMS
    )

    # Compute MFCC features
    mfcc = mfcc_step(wav.squeeze(0))

    # Compute delta (first derivative) and delta-delta (second derivative)
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta_delta = torchaudio.functional.compute_deltas(delta)

    # Compute Zero-Crossing Rate (ZCR)
    zcr = compute_zero_crossing_rate(wav, frame_length=2048, hop_length=256)  # Shape: (1, time_steps)
        
    # Take median across time (to match MFCC processing)
    zcr_median = torch.nanmedian(zcr, dim=1).values  

    # Compute Fundamental Frequency (F0)
    f0 = torchaudio.functional.detect_pitch_frequency(wav, sampling_rate,freq_high=300)

    # Apply log transformation to F0 to normalize gender differences in pitch.
    f0_log = torch.log(f0)
    
    # Take median of F0 across time
    f0_median = torch.nanmedian(f0_log, dim=0).values

    # Round all features
    mfcc =  mfcc.round(decimals=3)
    delta =  delta.round(decimals=3)
    delta_delta =  delta_delta.round(decimals=3)
    zcr_median = zcr_median.round(decimals=3)  
    f0_median = f0_median.round(decimals=3)

    # Combine MFCC, Delta, Delta-Delta, ZCR, and F0
    combined_features = torch.cat((mfcc, delta, delta_delta,), dim=0)  # Shape: (3 * n_mfcc, time_steps)

    # Transpose tensor to have time steps as rows
    combined_features = torch.transpose(combined_features, 0, 1)  # Shape: (time_steps, 4 * n_mfcc)

    # Replace 0. with NaN to avoid skewing statistics
    combined_features[combined_features == 0.] = float('nan')

    # Compute median of each feature across time
    final_features, _ = combined_features.nanmedian(dim=0, keepdim=False)

    # Convert to numpy array and aggregate
    result = np.hstack((result, final_features.numpy(), zcr_median.numpy(), f0_median.numpy()))

    return result

log("=== AUDIO PREPROCESSING STEPS ===")
log("1. Load the audio file.")
log("2. Convert stereo audio to mono (if applicable).")
log(f"3. Resample the audio to a target sampling rate ({SAMPLING_RATE} Hz).")
log("4. Apply amplitude normalization to scale audio between -1 and 1.")
log("5. Crop or pad the audio to a fixed length (4 seconds).")
log("=================================\n")

def preprocess_audio(file_path, target_sr=16000, max_length=4.0):
    """
    Preprocesses an audio file to standardize it for an emotional recognition system (SER).
    
    # Input:
    - file_path: Path to the audio file (.wav);
    - target_sr: Target sampling rate (e.g., 8 kHz);
    - max_length: Maximum length in seconds (cropping/padding);
    
    # Output:
    - processed_audio: Preprocessed audio tensor;
    - sr: Actual sampling rate.
    """
    # 1. Load audio
    wav, sr = torchaudio.load(file_path)

    # 2. Convert to mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # 3. Resampling
    if sr != target_sr:
        resample_step = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resample_step(wav)
        sr = target_sr

    # 4. Amplitude normalization
    wav = wav / wav.abs().max()

    # 5. Cropping/Padding
    # Calculate the target number of samples for max_length
    target_samples = int(max_length * target_sr)
    if wav.size(1) > target_samples:  # Cropping
        wav = wav[:, :target_samples]
    elif wav.size(1) < target_samples:  # Padding
        pad_length = target_samples - wav.size(1)
        wav = torch.nn.functional.pad(wav, (0, pad_length), mode='constant', value=0)

    return wav.squeeze(0), sr

# Load the data and extract features for each sound file
def load_data():

    # initialize empty Python lists
    x,y,u,t=[],[],[],[]
    
    # list all .wav files in the folder
    files = glob.glob(os.path.sep.join([f'./input/datasets/{DATASET}/audio/*.wav']))

    evaluations_df = pd.read_csv(os.path.sep.join([
                                            f'./input/datasets/{DATASET}/metadata/evaluations.csv']), header=0, dtype='str')

    # pre-process each audio file by "torchaudio"                                                                 
    for file in files:
        
        file_name = os.path.basename(file)
        complete_file_name = str(file_name)
        
        # Check if the file exists in evaluations_df
        matching_rows = evaluations_df.loc[evaluations_df['file_name'] == file_name, 'emotion_expressed']
        
        # add 'and matching_rows.values[0] != 'calm' ' to preprocess 7 emotions, otherwise 6
        if not matching_rows.empty:
            # If the file exists, get the emotion
            emotion = matching_rows.values[0]
            feature = extract_feature(file_name=file, sampling_rate=SAMPLING_RATE)
            x.append(feature)
            y.append(emotion)
            u.append(complete_file_name)
            t.append(evaluations_df.loc[evaluations_df['file_name'] == file_name, 'actor'].values[0]) 
        
    return  np.column_stack((x, y, u, t)) 

## when running on LAPTOP
if run_modality.iloc[0, 0] == 'testing':

    # merge all columns into a single DataFrame
    whole_data_set = pd.DataFrame(load_data())

    # rename columns
    # Get the total number of columns
    num_cols = whole_data_set.shape[1]

    # Map the last three columns to new names
    whole_data_set.rename(columns={num_cols - 3: "emotions", 
                                num_cols - 2: "file_name", 
                                num_cols - 1: "subject_serial_number"}, 
                        inplace=True)


    # save train-set + test-set as .csv files
    whole_data_set.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                                ('processed_data/talking_about_7_emotions_median.csv')]), index= False)


# when running on SERVER (assuming you have alraedy computed MFFCs (they are saved already) 
# in "talking_about_7_emotions_median.csv" for convenience. Skipping the MFCCs saves time, among training interation. Indeed, 
# they are the same for each training iteration)
elif run_modality.iloc[0, 0] == 'production':

    # Load the dataset
    whole_data_set = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                                    ('processed_data/talking_about_7_emotions_median.csv')]), header=0)

# Use the entire dataset as training set (will be split in train/test set later)
train_set_tmp = whole_data_set.copy()

# size of each set
overall_count = pd.unique(whole_data_set.loc[:, "subject_serial_number"]).size
train_set_count = pd.unique(train_set_tmp.loc[:, "subject_serial_number"]).size

log("=== PREPROCESSING FINISHED ===\n")

log("=== Dataset Class Distributions ===", False)

log(f"dataset size: {len(whole_data_set)}")
train_dist = whole_data_set['emotions'].value_counts()
train_dist_perc = whole_data_set['emotions'].value_counts(normalize=True) * 100

train_summary = "\n".join([f"{cls}: {count} ({perc:.2f}%)" for cls, count, perc in zip(train_dist.index, train_dist.values, train_dist_perc.values)])
log(f"dataset summary:\n{train_summary}", False)

# count single IDs
log('total ID count: {}'.format(overall_count))

log('train_set ID count: {}'.format(train_set_count))

log('percentage train_set: {}'.format((train_set_count)/overall_count))

# pop dependent variable
y_train = train_set_tmp.pop("emotions")

# pop file_name column
file_name_series_train = train_set_tmp.pop("file_name")

# pop subject_serial_number
subject_serial_number_train = train_set_tmp.pop("subject_serial_number")

# deep copy
x_train = train_set_tmp.copy()

# concatenate DataFrame + Series
train_set = pd.concat([pd.DataFrame(x_train).reset_index(drop=True),                               
                       y_train.reset_index(drop=True), 
                       file_name_series_train.reset_index(drop=True),
                       subject_serial_number_train.reset_index(drop=True)],
                               ignore_index = True, 
                               axis = 1)

# Get the total number of columns
num_cols = whole_data_set.shape[1]

# Map the last three columns to new names
train_set.rename(columns={num_cols - 3: "emotions", 
                               num_cols - 2: "file_name", 
                               num_cols - 1: "subject_serial_number"}, 
                      inplace=True)

# save train-set as .csv files
train_set.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                           ('processed_data/preprocessed_set.csv')]), index= False)


# end time according to computer clock
end_time = time.time()

# calculate total execution time
total_execution_time = pd.Series(np.round((end_time - start_time), 2)).rename('total_training_runtime_seconds')

# shows run-time's timestamps + total execution time
log(f"End time (Unix timestamp): {end_time}")
log(f"Total execution time (seconds): {total_execution_time}")
log("=================================\n")

# Delete large objects to free memory
del whole_data_set, train_set, train_set_tmp, x_train, y_train

# Force garbage collection for CPU memory
gc.collect()