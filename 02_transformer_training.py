"""
TITLE: "Speech Emotion Recognition (SER) using Transformers"
AUTHORS: Giuseppe Lentini
LAST UPDATE: 2025-07-11
PYTHON VERSION: 3.11.11
"""

###############################################################################
## IMPORTING LIBRARIES
import torch
import torchaudio
import pandas as pd
import numpy as np
import os
import gc
import random
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
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import recall_score, accuracy_score, f1_score
import warnings 
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

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
MODEL_NAME = "openai/whisper-small"
BATCH_SIZE = 32
EPOCHS = 13
LEARNING_RATE = 3e-5
label_mapping = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Joy': 3, 'Neutral': 4, 'Sadness': 5, 'Surprise': 6}
NUM_LABELS = len(label_mapping)
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
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.05)

    config = WhisperConfig.from_pretrained(MODEL_NAME)
    config.num_labels = NUM_LABELS
    config.label2id = label_mapping
    config.id2label = {v: k for k, v in label_mapping.items()}

    config.max_length = None
    config.suppress_tokens = None
    config.begin_suppress_tokens = None

    # Load and configure the model with the suggested hyperparameters
    model = WhisperForAudioClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True
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
# ## HYPERPARAMETERS TUNING 
# processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# data_collator = DataCollatorWithPadding(processor)

# if(os.path.exists('output/optuna_trial/best_params.csv')):
#     best_params_df = pd.read_csv('output/optuna_trial/best_params.csv')
# else:
#     # Get a well-balanced train/validation split
#     train_df, val_df, _ = get_best_split(df_data, NUMBER_SPLIT_ROUNDS, SEED)
#     train_dataset = Dataset.from_pandas(train_df).map(preprocess_function, batched=True)
#     val_dataset = Dataset.from_pandas(val_df).map(preprocess_function, batched=True)
    
#     print("\n|START HYPERPARAMETER TUNING|")
    
#     study = optuna.create_study(
#         direction="maximize",
#         sampler=TPESampler(seed=SEED),
#         pruner=MedianPruner(
#             n_startup_trials=3,
#             n_warmup_steps=2,
#             interval_steps=1
#         )
#     )
#     study.optimize(lambda trial: objective(trial, train_dataset, val_dataset), n_trials=60, show_progress_bar=True)
#     best_params = study.best_params
#     best_params_df = pd.DataFrame([study.best_params])
#     best_params_df["best_value"] = study.best_value  
#     best_params_df.to_csv("output/optuna_trial/best_params.csv", index=False)

###############################################################################
## START ITERATIVE TRAINING

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
metric = evaluate.load("accuracy")
data_collator = DefaultDataCollator(return_tensors="pt")

training_args = TrainingArguments(
    output_dir="./whisper/ser_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="uar",
    greater_is_better=True,

    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.1,
    weight_decay=0.05,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-8,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,

    num_train_epochs=EPOCHS,
    logging_dir="./logs",
    logging_steps=50,

    fp16=True,
    gradient_checkpointing=True,

    dataloader_num_workers=16,
    optim="adamw_torch",
    ddp_find_unused_parameters=False,

    max_grad_norm=1.0,
    label_smoothing_factor=0.05,
)


additional_test_dataset = Dataset.from_pandas(df_additional_test)
additional_test_dataset = additional_test_dataset.map(preprocess_function, batched=True)

for i in range(15):
    print(f"Starting training MODEL_{i}")

    folder_path = f"./output/MODEL_{i}"
    os.makedirs(folder_path, exist_ok=True)

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
    )

    # clean memory before training
    gc.collect()
    torch.cuda.empty_cache()

    trainer.train()

    trainer.save_model(f"{folder_path}/ser_finetuned_model")
    feature_extractor.save_pretrained(f"{folder_path}/ser_finetuned_model")

    metrics = trainer.evaluate()
    output_file = f"{folder_path}/training_metrics.txt"

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