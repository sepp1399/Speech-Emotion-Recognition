"""
TITLE: "Speech Emotion Recognition (SER) using Transformers"
AUTHORS: Giuseppe Lentini
LAST UPDATE: 2025-03-28
PYTHON VERSION: 3.11.11
"""

###############################################################################
## IMPORTING LIBRARIES
import torch
import torchaudio
import pandas as pd
import numpy as np
import os
import evaluate
from sklearn.model_selection import StratifiedGroupKFold
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from transformers import DataCollatorWithPadding
import gc

###############################################################################
## SET PARAMETERS

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
PROCESSED_DATA_PATH = "processed_data.parquet"
PROCESSED_EMOVO_PATH = "processed_emovo.parquet"

DATA_FOLDER = "input/datasets/Emozionalmente_augmented_core"
AUDIO_FOLDER = os.path.join(DATA_FOLDER, "audio/")
CSV_FILE = os.path.join(DATA_FOLDER, "metadata", "evaluations.csv")

EMOVO_FOLDER = "input/datasets/EMOVO"
EMOVO_AUDIO_FOLDER = os.path.join(EMOVO_FOLDER, "audio/")
EMOVO_CSV_FILE = os.path.join(EMOVO_FOLDER, "metadata", "evaluations.csv")

# Main parameters
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
NUM_LABELS = 7
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 5e-5
MAX_DURATION = 4  # in seconds

###############################################################################
## METHODS

def get_best_split(df, max_attempts=1000, seed=42):
    """
    Finds the best balanced split of a dataset using StratifiedGroupKFold.
    
    The function iterates multiple times with different random seeds to find the split 
    that minimizes class imbalance across training, validation, and test sets.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        max_attempts (int, optional): The number of different random states to try. Default is 1000.
        seed (int, optional): Seed for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing three DataFrames (train_df, val_df, test_df) corresponding 
               to the best-balanced split.
    """
    best_balance = float('inf')  # Initialize the best balance score to infinity
    best_splits = None  # Store the best split configuration
    np.random.seed(seed)  # Set seed for reproducibility
    random_states = np.random.randint(0, 10000, size=max_attempts)  # Generate random seeds for each attempt

    for random_state in random_states:
        # First split: 5-fold stratified group split
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        splits = list(sgkf.split(df, df["label"], groups=df["group"]))
        
        train_idx, temp_idx = splits[0]  # Use the first fold for training
        val_test_data = df.iloc[temp_idx].reset_index(drop=True)  # Remaining data for validation and testing

        # Second split: Further split the remaining data into validation and test sets
        sgkf_val_test = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
        val_idx, test_idx = list(sgkf_val_test.split(val_test_data, val_test_data["label"], groups=val_test_data["group"]))[0]

        # Create DataFrames for the train, validation, and test sets
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = val_test_data.iloc[val_idx].reset_index(drop=True)
        test_df = val_test_data.iloc[test_idx].reset_index(drop=True)

        # Compute class balance by measuring the standard deviation of label distributions
        balance_score = (
            train_df["label"].value_counts(normalize=True).std() +  # Train class distribution variance
            val_df["label"].value_counts(normalize=True).std() +    # Validation class distribution variance
            test_df["label"].value_counts(normalize=True).std()     # Test class distribution variance
        )

        print(f"random state: {random_state}, final balance_score: {balance_score}")

        # Update best split if the current one has better balance
        if balance_score < best_balance:
            best_balance = balance_score
            best_splits = (train_df, val_df, test_df)

    return best_splits  # Return the most balanced split

def preprocess_function(batch):
    """
    Preprocesses a batch of audio files by performing the following steps:
    - Loads the audio from file paths.
    - Converts stereo audio to mono.
    - Resamples audio to 16kHz if needed.
    - Pads or truncates the waveform to a fixed length.
    - Converts the processed audio into tensors for model input.
    - Moves the processed data to the GPU for efficient computation.

    Args:
        batch (dict): A dictionary containing:
            - "path" (list of str): Paths to the audio files.
            - "label" (list of int): Corresponding labels for classification.

    Returns:
        dict: The updated batch with processed audio inputs and labels, ready for model training or inference.
    """
    max_length = MAX_DURATION * 16000  # Define the maximum length in samples (assuming MAX_DURATION is in seconds)
    audio_arrays = []  # List to store processed audio waveforms
    
    for path in batch["path"]:
        waveform, sample_rate = torchaudio.load(path)  # Load the audio file

        # Convert stereo to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample the audio to 16kHz if it's not already at that sample rate
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Move the waveform to the GPU for processing
        waveform = waveform.to(device)

        # Apply padding or truncation to ensure a fixed length
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]  # Truncate if too long
        else:
            padding = max_length - waveform.shape[1]  # Calculate required padding
            waveform = torch.nn.functional.pad(waveform, (0, padding))  # Apply padding at the end

        # Convert waveform to a NumPy array and move it to the CPU for further processing
        audio_arrays.append(waveform.squeeze(0).cpu().numpy())

    # Convert the processed audio into input tensors using the processor
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Store input values and labels in the batch dictionary, moving them to the GPU
    batch["input_values"] = inputs.input_values.to(device)
    batch["labels"] = torch.tensor(batch["label"]).to(device)
    
    return batch  # Return the processed batch

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
    logits, labels = eval_pred  # Unpack model outputs and labels
    predictions = np.argmax(logits, axis=-1)  # Get the predicted class with the highest probability
    return metric.compute(predictions=predictions, references=labels)  # Compute and return evaluation metrics

###############################################################################
## PREPROCESSING

if os.path.exists(PROCESSED_DATA_PATH) and os.path.exists(PROCESSED_EMOVO_PATH):
    print("Processed data found! Loading...")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    df_emovo = pd.read_parquet(PROCESSED_EMOVO_PATH)
else:
    print("No processed data found. Processing...")
    df = pd.read_csv(CSV_FILE)
    df = df.rename(columns={"file_name": "path", "emotion_expressed": "label", "actor": "group"})  
    df["path"] = df["path"].apply(lambda x: os.path.join(AUDIO_FOLDER, x))
    label_mapping = {label: i for i, label in enumerate(df["label"].unique())}
    print(label_mapping)
    df["label"] = df["label"].map(label_mapping)
    df.to_parquet(PROCESSED_DATA_PATH)
    
    df_emovo = pd.read_csv(EMOVO_CSV_FILE)
    df_emovo = df_emovo.rename(columns={"file_name": "path", "emotion_expressed": "label"})
    df_emovo["path"] = df_emovo["path"].apply(lambda x: os.path.join(EMOVO_AUDIO_FOLDER, x))
    df_emovo["label"] = df_emovo["label"].map(label_mapping)
    df_emovo.to_parquet(PROCESSED_EMOVO_PATH)


###############################################################################
## MODEL

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    classifier_proj_size=256,
).to(device)  # Move model to GPU

###############################################################################
## TRAINING 

training_args = TrainingArguments(
    output_dir="./ser_model",

    # Evaluation & Saving
    evaluation_strategy="epoch",  # Evaluate at each epoch
    save_strategy="epoch",  # Save the model at each epoch
    save_total_limit=2,  # Keep only the 2 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Optimization
    learning_rate=LEARNING_RATE,  # Lower learning rate for stability
    lr_scheduler_type="cosine",  # Cosine decay for smoother training
    warmup_ratio=0.1,  # 10% of steps for warm-up
    weight_decay=0.01,  # Avoid overfitting
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
    optim="adamw_hf",  # Advanced optimizer

    # Distributed Training
    ddp_find_unused_parameters=False,  # Optimized for DDP
)

metric = evaluate.load("accuracy")

data_collator = DataCollatorWithPadding(processor)

###############################################################################
## START ITERATIVE TRAINING

for i in range(15):
    print(f"Starting training MODEL_{i}")

    folder_path = f"./output/MODEL_{i}"
    os.makedirs(folder_path, exist_ok=True)

    # Load a new model on GPU
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        classifier_proj_size=256,
    ).to(device)

    # Dataset
    train_df, val_df, test_df = get_best_split(df, 1000, i)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    emovo_dataset = Dataset.from_pandas(df_emovo)

    # Preprocessing on GPU
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    emovo_dataset = emovo_dataset.map(preprocess_function, batched=True)

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

    test_results_emovo = trainer.evaluate(emovo_dataset)
    print(f"Test EMOVO Accuracy: {test_results_emovo['eval_accuracy']:.4f}")

    output_csv_path = os.path.join('output', "performance.csv")

    if not os.path.exists(output_csv_path):
        columns = ["Model", "Test accuracy", "EMOVO accuracy"]
        results_df = pd.DataFrame(columns=columns)
        results_df.to_csv(output_csv_path, index=False)
    else:
        results_df = pd.read_csv(output_csv_path)

    new_row = pd.DataFrame({
        "Model": [f"MODEL_{i}"],
        "Test accuracy": [round(test_results['eval_accuracy'], 4)],
        "EMOVO accuracy": [round(test_results_emovo['eval_accuracy'], 4)]
    })

    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(output_csv_path, index=False)

    # clean memory
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
