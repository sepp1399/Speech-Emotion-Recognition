"""
TITLE: "Data Augmentation Finder for Speech Emotion Recognition"
DESCRIPTION: This script evaluates various data augmentation techniques for speech emotion recognition using a Wav2Vec2 model. It finds the most balanced dataset split and trains models on augmented data, measuring their performance.
AUTHORS: Giuseppe Lentini
LAST UPDATE: 2025-06-19
PYTHON VERSION: 3.11.11
"""

###############################################################################
## IMPORT LIBRARIES
import random
import torch
import torchaudio
import pandas as pd
import numpy as np
import evaluate
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    WhisperConfig,
    WhisperForAudioClassification,
    WhisperFeatureExtractor,
    DefaultDataCollator
)
from datasets import Dataset
from transformers import DataCollatorWithPadding
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
from itertools import combinations
from transformers.utils import logging
import os
from sklearn.metrics import recall_score, accuracy_score, f1_score

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.set_verbosity_error()

SEED = 13

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

###############################################################################
## DEFINE GLOBAL PARAMETERS

# Choose device: use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Dataset paths
TEST_DATASET = "Multilinguis"
TEST_BASE_DIR = f"input/datasets/{TEST_DATASET}"
TEST_AUDIO_DIR = os.path.join(TEST_BASE_DIR, "audio")
TEST_METADATA_PATH = os.path.join(TEST_BASE_DIR, "metadata", "evaluations.csv")

# Model & training parameters
MODEL_NAME = "openai/whisper-small"
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 7
LEARNING_RATE = 3e-5
MAX_AUDIO_DURATION_SECONDS = 4
TARGET_SAMPLE_RATE = 16000
NUM_REPEATED_TRAININGS = 3
NUMBER_SPLIT_ROUNDS = 3000

AUGMENTED_DATA_DIR = "input/augmentation"
EVALUATION_RESULTS_PATH = "output/best_techniques.csv"
fixed_technique = "emozionalmente_core"
label_mapping = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Joy': 3, 'Neutral': 4, 'Sadness': 5, 'Surprise': 6}
NUM_LABELS = len(label_mapping)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
## FUNCTIONS

def get_split(random_state, df_data, max_attempts):
    """
    Perform single split and calculate balance score.
    """
    # First split: 5-fold stratified group split
    best_balance = float('inf')  # Initialize the best balance score to infinity
    best_splits = None  # Store the best split configuration
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = list(sgkf.split(df_data, df_data["label"], groups=df_data["group"]))
    
    train_idx, temp_idx = splits[0]  # Use the first fold for training
    val_test_data = df_data.iloc[temp_idx].reset_index(drop=True)  # Remaining data for validation and testing

    # Second split: Further split the remaining data into validation and test sets
    for i in range(100):
        sgkf_val_test = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=i)
        val_idx, test_idx = list(sgkf_val_test.split(val_test_data, val_test_data["label"], groups=val_test_data["group"]))[0]

        # Create DataFrames for the train, validation, and test sets
        train_df = df_data.iloc[train_idx].reset_index(drop=True)
        val_df = val_test_data.iloc[val_idx].reset_index(drop=True)
        test_df = val_test_data.iloc[test_idx].reset_index(drop=True)

        # Compute class balance by measuring the standard deviation of label distributions
        balance_score = (
            train_df["label"].value_counts(normalize=True).std() +  # Train class distribution variance
            val_df["label"].value_counts(normalize=True).std() +    # Validation class distribution variance
            test_df["label"].value_counts(normalize=True).std()     # Test class distribution variance
        )

        if balance_score < best_balance:
            best_balance = balance_score
            best_splits = (train_df, val_df, test_df)

    # Compute class balance by measuring the standard deviation of label distributions
    balance_score = (
        train_df["label"].value_counts(normalize=True).std() +  # Train class distribution variance
        val_df["label"].value_counts(normalize=True).std() +    # Validation class distribution variance
        test_df["label"].value_counts(normalize=True).std()     # Test class distribution variance
    )

    return balance_score, best_splits

def get_best_split(df_data, max_attempts=1000, seed=42, n_jobs=-1):
    """
    Finds the best balanced split of a dataset using StratifiedGroupKFold, in parallel.

    Args:
        df_data (pd.DataFrame): The input DataFrame containing the data.
        max_attempts (int, optional): The number of different random states to try. Default is 1000.
        seed (int, optional): Seed for reproducibility. Default is 42.
        n_jobs (int, optional): Number of jobs to run in parallel. Default is -1 (use all available cores).

    Returns:
        tuple: A tuple containing three DataFrames (train_df, val_df, test_df) corresponding 
               to the best-balanced split.
    """
    best_balance = float('inf')  # Initialize the best balance score to infinity
    best_splits = None  # Store the best split configuration
    np.random.seed(seed)  # Set seed for reproducibility
    random_states = np.random.randint(0, 10000, size=max_attempts)  # Generate random seeds for each attempt

    with tqdm(total=max_attempts, desc="Splitting progress (wait...)") as pbar:
        # Parallelize the execution of multiple splits
        results = Parallel(n_jobs=n_jobs)(
            delayed(get_split)(random_state, df_data, max_attempts) for random_state in random_states
        )

        # Raccolta dei risultati e stampa alla fine
        for i, (balance_score, partial_splits) in enumerate(results):
            pbar.update(1)

            print(f"Random state {random_states[i]}: Balance score = {balance_score}")

            # Update best split if the current one has better balance
            if balance_score < best_balance:
                best_balance = balance_score
                best_splits = partial_splits

    return best_splits  # Return the most balanced split

def preprocess_function(batch):
    audio_arrays = []

    for path in batch["path"]:

        waveform, sample_rate = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        audio_arrays.append(waveform.squeeze().numpy())

    inputs = feature_extractor(audio_arrays, sampling_rate=16000)
    batch["input_features"] = inputs["input_features"]
    batch["labels"] = batch["label"]
    return batch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "uar": recall_score(labels, preds, average="macro"),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


def load_test_dataset():
    """
    Load test dataset and standardize column names and labels.
    """
    print(f"Loading {TEST_DATASET} dataset...")
    df = pd.read_csv(TEST_METADATA_PATH)
    df = df.rename(columns={"file_name": "path", "emotion_recognized": "label", "actor": "group"})
    df["path"] = df["path"].apply(lambda x: os.path.join(TEST_AUDIO_DIR, x))
    label_to_index = {label: i for i, label in enumerate(sorted(df["label"].unique()))}
    df["label"] = df["label"].map(label_to_index)
    return df

def load_augmented_combination(combination: str):
    print(f"\nLoading augmented data: {combination}")
    all_dfs = []

    for technique in combination.split("+"):
        technique_dir = os.path.join(AUGMENTED_DATA_DIR, technique)
        technique_csv = os.path.join(technique_dir, "metadata/evaluations.csv")

        if not os.path.exists(technique_csv):
            print(f"Warning: CSV for '{technique}' not found at {technique_csv}")
            continue

        df = pd.read_csv(technique_csv)
        df = df.rename(columns={"file_name": "path", "emotion_expressed": "label", "actor": "group"})
        df["path"] = df["path"].apply(lambda x: os.path.join(technique_dir, "audio", x))
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid data loaded for selected combination.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    label_map = {label: i for i, label in enumerate(sorted(combined_df["label"].unique()))}
    combined_df["label"] = combined_df["label"].map(label_map)

    return combined_df

###############################################################################
## PREPARE COMBINATIONS TO EVALUATE

if os.path.exists(EVALUATION_RESULTS_PATH):
    print(f"Found existing evaluation file at {EVALUATION_RESULTS_PATH}. Loading.")
    evaluation_table = pd.read_csv(EVALUATION_RESULTS_PATH)
else:
    print("Generating new combination table...")

    available_techniques = random.sample([
        name for name in os.listdir(AUGMENTED_DATA_DIR)
        if os.path.isdir(os.path.join(AUGMENTED_DATA_DIR, name)) and not name.startswith('.')
    ], k=len(os.listdir(AUGMENTED_DATA_DIR)) - 2)

    combination_data = []

    for r in range(5, len(available_techniques) + 1):
        for combo in combinations(available_techniques, r):
            if fixed_technique in combo:
                combination_data.append({
                    "techniques": "+".join(combo),
                    "model_0": None,
                    "model_1": None,
                    "model_2": None,
                    "avg": None
                })

    evaluation_table = pd.DataFrame(combination_data)
    evaluation_table.to_csv(EVALUATION_RESULTS_PATH, index=False)
    print(f"Combinations saved to {EVALUATION_RESULTS_PATH}")

###############################################################################
## INITIALIZE MODEL COMPONENTS

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
data_collator = DefaultDataCollator(return_tensors="pt")

training_config = TrainingArguments(
    output_dir="./ser_model",
    
    eval_strategy="epoch",
    save_strategy="no",
    save_total_limit=0,
    load_best_model_at_end=False,
    metric_for_best_model="uar",
    
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.1,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-8,
    
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    num_train_epochs=EPOCHS,
    
    disable_tqdm=False,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50,
    
    fp16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    optim="adamw_torch",
    
    ddp_find_unused_parameters=False,
    max_grad_norm=1.0,
    label_smoothing_factor=0.05,
)

###############################################################################
## TRAIN MODELS FOR EACH COMBINATION

test_df = load_test_dataset()
test_dataset = Dataset.from_pandas(test_df).map(preprocess_function, batched=True)

for index, row in evaluation_table.iterrows():
    if not pd.isna(row['avg']):
        continue

    technique_combo = row["techniques"]
    augmented_df = load_augmented_combination(technique_combo)

    for run in range(NUM_REPEATED_TRAININGS):

        if not pd.isna(row[f'model_{run}']):
            continue

        print(f'Processing model_{run}')

        seed = 0
        train_df, val_df, _ = get_best_split(augmented_df, max_attempts=NUMBER_SPLIT_ROUNDS, seed=seed)

        train_ds = Dataset.from_pandas(train_df).map(preprocess_function, batched=True)
        val_ds = Dataset.from_pandas(val_df).map(preprocess_function, batched=True)

    # Load a new model on GPU
        config = WhisperConfig.from_pretrained(MODEL_NAME)
        config.num_labels = NUM_LABELS
        config.label2id = label_mapping
        config.id2label = {v: k for k, v in label_mapping.items()}

        config.max_length = None
        config.suppress_tokens = None
        config.begin_suppress_tokens = None

        model = WhisperForAudioClassification.from_pretrained(
            MODEL_NAME,
            config=config,
            ignore_mismatched_sizes=True
        ).to(device)

        trainer = Trainer(
            model=model,
            args=training_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n|{technique_combo}| Training model run {run}")
        trainer.train()

        evaluation_metrics = trainer.evaluate(test_dataset)
        evaluation_table.at[index, f"model_{run}"] = round(evaluation_metrics['eval_uar'], 4)
        evaluation_table.to_csv(EVALUATION_RESULTS_PATH, index=False)
         
        del model, trainer, train_ds, val_ds, train_df, val_df
        gc.collect()
        torch.cuda.empty_cache()

        seed += 1

        if run == NUM_REPEATED_TRAININGS - 1:
            average_accuracy = evaluation_table.loc[index, ["model_0", "model_1", "model_2"]].mean()
            evaluation_table.at[index, "avg"] = round(average_accuracy, 4)
            evaluation_table.to_csv(EVALUATION_RESULTS_PATH, index=False)
            print(f"Finished {technique_combo}. Average accuracy: {average_accuracy:.4f}")