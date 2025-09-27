"""
TITLE: "Speech Emotion Recognition (SER) using Transformers"
AUTHORS: Giuseppe Lentini
LAST UPDATE: 2025-06-26
PYTHON VERSION: 3.11.11
"""

###############################################################################
## IMPORTING LIBRARIES
import torch
import torchaudio
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold
from transformers import Wav2Vec2Processor,WavLMForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from transformers import DataCollatorWithPadding
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
import random
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import recall_score, accuracy_score, f1_score
from transformers import EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback
import warnings

warnings.filterwarnings("ignore")

SEED = 13

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
###############################################################################
## SET PARAMETERS

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
PROCESSED_DATA_PATH = "processed_data.parquet"
PROCESSED_TEST_PATH = "processed_test.parquet"

TEST_DATASET = "Multilinguis"
TEST_FOLDER = f"input/datasets/{TEST_DATASET}"
TEST_AUDIO_FOLDER = os.path.join(TEST_FOLDER, "audio/")
TEST_CSV_FILE = os.path.join(TEST_FOLDER, "metadata", "evaluations.csv")

AUGMENTED_DATA_DIR = "input/augmentation"
TECHNIQUES_CSV_PATH = "output/best_techniques.csv"

NUMBER_SPLIT_ROUNDS = 3000

# Main parameters
MODEL_NAME = "jonatasgrosman/exp_w2v2t_it_wavlm_s895"
NUM_LABELS = 7
BATCH_SIZE = 32
EPOCHS = 13
LEARNING_RATE = 3e-5
MAX_DURATION = 4  # in seconds

###############################################################################
## METHODS

def get_split(random_state, df_data):
    """
    Perform single split and calculate balance score.
    """
    # First split: 5-fold stratified group split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = list(sgkf.split(df_data, df_data["label"], groups=df_data["group"]))
    
    train_idx, temp_idx = splits[0]  # Use the first fold for training
    val_test_data = df_data.iloc[temp_idx].reset_index(drop=True)  # Remaining data for validation and testing

    # Second split: Further split the remaining data into validation and test sets
    sgkf_val_test = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
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

    return balance_score, train_df, val_df, test_df

def get_best_split(df_data, max_attempts=1000, seed=42, n_jobs=6):
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
            delayed(get_split)(random_state, df_data) for random_state in random_states
        )

        # Raccolta dei risultati e stampa alla fine
        for i, (balance_score, train_df, val_df, test_df) in enumerate(results):
            pbar.update(1)

            print(f"Random state {random_states[i]}: Balance score = {balance_score}")

            # Update best split if the current one has better balance
            if balance_score < best_balance:
                best_balance = balance_score
                best_splits = (train_df, val_df, test_df)

    return best_splits  # Return the most balanced split

def preprocess_function(batch):
    """
    Preprocesses a batch of audio files for model input, compatible with datasets.map(batched=True).
    Returns numpy arrays instead of tensors.
    """
    max_length = MAX_DURATION * 16000  # lunghezza fissa
    audio_arrays = []

    for path in batch["path"]:
        waveform, sample_rate = torchaudio.load(path)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample a 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Padding o truncation
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        else:
            padding = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        audio_arrays.append(waveform.squeeze(0).numpy())  # torna su CPU e numpy

    # Tokenization
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)

    return {
        "input_values": inputs["input_values"],
        "attention_mask": inputs.get("attention_mask"),
        "labels": np.array(batch["label"])
    }

def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for a classification model.

    This function takes model predictions and ground truth labels, extracts the most 
    probable class for each prediction, and computes evaluation metrics such as 
    accuracy, precision, recall, or F1-score using a predefined metric function.

    Args:
        eval_pred (tuple): A tuple containing:
            - logits (np.ndarray): The model's raw output predictions (logits).
            - labels (np.ndarray): The ground truth labels.

    Returns:
        dict: A dictionary containing the computed evaluation metrics.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "uar": recall_score(labels, preds, average="macro"),
        "macro_f1": f1_score(labels, preds, average="macro")
    }

def load_augmented_combination(combination: str):
    print(f"\nLoading augmented data: {combination}")
    all_dfs = []

    for technique in combination.split("+"):
        technique_dir = os.path.join(AUGMENTED_DATA_DIR, technique)
        technique_csv = os.path.join(technique_dir, "metadata/evaluations.csv")

        if not os.path.exists(technique_csv):
            print(f"Warning: CSV for '{technique}' not found at {technique_csv}")
            continue

        df_data = pd.read_csv(technique_csv)
        df_data = df_data.rename(columns={"file_name": "path", "emotion_expressed": "label", "actor": "group"})
        df_data["path"] = df_data["path"].apply(lambda x: os.path.join(technique_dir, "audio", x))
        all_dfs.append(df_data)

    if not all_dfs:
        raise ValueError("No valid data loaded for selected combination.")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    return combined_df

# Define the objective function for Optuna hyperparameter optimization
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        val_uar = metrics.get("eval_uar")
        if val_uar is None:
            return

        self.trial.report(val_uar, step=state.epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
            
def objective(trial, train_dataset, val_dataset):
    
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    classifier_proj_size = trial.suggest_categorical("classifier_proj_size", [256, 512])
    hidden_dropout = trial.suggest_float("hidden_dropout", 0.05, 0.2)
    attention_dropout = trial.suggest_float("attention_dropout", 0.05, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.05)

    # Load and configure the model with the suggested hyperparameters
    model = WavLMForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        classifier_proj_size=classifier_proj_size,
    ).to(device)

    # Define dynamic training arguments based on trial suggestions
    dynamic_args = TrainingArguments(
        output_dir="./output/optuna_trial",  # Directory to store outputs (not saving checkpoints here)
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        eval_strategy="epoch",  # Evaluate once per epoch
        save_strategy="no",  # Disable checkpoint saving during tuning
        logging_steps=50,
        fp16=True,  # Enable mixed precision for faster training
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        dataloader_num_workers=2,
        optim="adamw_torch",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=dynamic_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[OptunaPruningCallback(trial)],
    )

    # Train the model and evaluate on validation set
    trainer.train()
    metrics = trainer.evaluate()
    val_uar = metrics["eval_uar"]

    del model, trainer, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return val_uar

###############################################################################
## PREPROCESSING

if os.path.exists(PROCESSED_DATA_PATH) and os.path.exists(PROCESSED_TEST_PATH):
    print("Processed data found! Loading...")
    df_data = pd.read_parquet(PROCESSED_DATA_PATH)
    df_additional_test = pd.read_parquet(PROCESSED_TEST_PATH)
else:
    print("No processed data found. Processing...")

    techniques_csv = pd.read_csv(TECHNIQUES_CSV_PATH)
    best_row = techniques_csv.loc[techniques_csv['avg'].idxmax()]
    best_technique = best_row['techniques']

    df_data = load_augmented_combination(best_technique)

    label_mapping = {label: i for i, label in enumerate(df_data["label"].unique())}
    print(label_mapping)
    df_data["label"] = df_data["label"].map(label_mapping)
    df_data.to_parquet(PROCESSED_DATA_PATH)

    df_additional_test = pd.read_csv(TEST_CSV_FILE)
    df_additional_test = df_additional_test.rename(columns={"file_name": "path", "emotion_recognized": "label"})
    df_additional_test["path"] = df_additional_test["path"].apply(lambda x: os.path.join(TEST_AUDIO_FOLDER, x))
    df_additional_test["label"] = df_additional_test["label"].map(label_mapping)
    df_additional_test.to_parquet(PROCESSED_TEST_PATH)

###############################################################################
## HYPERPARAMETERS TUNING 
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

data_collator = DataCollatorWithPadding(processor)

if(os.path.exists('output/optuna_trial/best_params.csv')):
    best_params_df = pd.read_csv('output/optuna_trial/best_params.csv')
else:
    # Get a well-balanced train/validation split
    train_df, val_df, _ = get_best_split(df_data, NUMBER_SPLIT_ROUNDS, SEED)
    train_dataset = Dataset.from_pandas(train_df).map(preprocess_function, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(preprocess_function, batched=True)
    
    print("\n|START HYPERPARAMETER TUNING|")
    
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2,
            interval_steps=1
        )
    )
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset), n_trials=60, show_progress_bar=True)
    best_params = study.best_params
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df["best_value"] = study.best_value  
    best_params_df.to_csv("output/optuna_trial/best_params.csv", index=False)

###############################################################################
## START ITERATIVE TRAINING

additional_test_dataset = Dataset.from_pandas(df_additional_test)
additional_test_dataset = additional_test_dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./ser_model",

    # Evaluation & Saving
    eval_strategy="epoch",  # Evaluate at each epoch
    save_strategy="epoch",  # Save the model at each epoch
    save_total_limit=2,  # Keep only the 2 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="uar",

    # Optimization
    learning_rate=best_params["learning_rate"],
    lr_scheduler_type="cosine",  # Cosine decay for smoother training
    weight_decay=best_params["weight_decay"],
    warmup_ratio=0.1,  # 10% of steps for warm-up
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-8,

    # Batch Size & Epochs
    per_device_train_batch_size=BATCH_SIZE,  # Adjust based on GPU memory
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,  # If batch size is too large for GPU
    num_train_epochs=EPOCHS,  # More epochs improve generalization in SER

    # Logging & Monitoring
    logging_dir="./logs",
    logging_steps=50,  # Log every 50 steps

    # Performance Boost
    fp16=True,  # Mixed Precision for faster training
    gradient_checkpointing=True,  # Reduce memory usage
    dataloader_num_workers=4,  # Speed up data loading
    optim="adamw_torch",  # Advanced optimizer

    # Distributed Training
    ddp_find_unused_parameters=False,  # Optimized for DDP
)

for i in range(15):
    print(f"Starting training MODEL_{i}")

    folder_path = f"./output/MODEL_{i}"
    os.makedirs(folder_path, exist_ok=True)

    # Load a new model on GPU
    model = WavLMForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        attention_dropout=best_params["attention_dropout"],
        hidden_dropout=best_params["hidden_dropout"],
        classifier_proj_size=best_params["classifier_proj_size"],
    ).to(device)  # Move model to GPU

    # Dataset
    train_df, val_df, test_df = get_best_split(df_data, NUMBER_SPLIT_ROUNDS, i)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Preprocessing on GPU
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    train_dataset.to_csv(folder_path + "/train_dataset.csv")
    val_dataset.to_csv(folder_path + "/val_dataset.csv")
    test_dataset.to_csv(folder_path + "/test_dataset.csv")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # clean memory before training
    gc.collect()
    torch.cuda.empty_cache()

    trainer.train()

    trainer.save_model(f"./output/MODEL_{i}/ser_finetuned_model")
    processor.save_pretrained(f"./output/MODEL_{i}/ser_finetuned_model")

    metrics = trainer.evaluate()
    output_file = f"./output/MODEL_{i}/training_metrics.txt"
    
    with open(output_file, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Metrics saved to {output_file}")

    test_results = trainer.evaluate(test_dataset)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")

    additional_test_results = trainer.evaluate(additional_test_dataset)
    print(f"Test {TEST_DATASET} Accuracy: {additional_test_results['eval_accuracy']:.4f}")

    output_csv_path = os.path.join('output', "performance.csv")

    if not os.path.exists(output_csv_path):
        columns = ["Model", "Test accuracy", f"{TEST_DATASET} accuracy"]
        results_df = pd.DataFrame(columns=columns)
        results_df.to_csv(output_csv_path, index=False)
    else:
        results_df = pd.read_csv(output_csv_path)

    new_row = pd.DataFrame({
        "Model": [f"MODEL_{i}"],
        "Test accuracy": [round(test_results['eval_accuracy'], 4)],
        f"{TEST_DATASET} accuracy": [round(additional_test_results['eval_accuracy'], 4)]
    })

    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(output_csv_path, index=False)

    # clean memory
    del model, trainer, train_dataset, val_dataset, test_dataset, results_df, new_row, train_df, val_df, test_df, metrics
    
    gc.collect()
    torch.cuda.empty_cache()