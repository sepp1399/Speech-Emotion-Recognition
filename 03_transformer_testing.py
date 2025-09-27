"""
TITLE: "Multi dataset test for SER models"
AUTHORS: Giuseppe Lentini
LAST UPDATE: 2025-04-02
PYTHON VERSION: 3.11.11
"""

###############################################################################
## IMPORTING LIBRARIES
import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn.functional as F
import onnxruntime as ort
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, matthews_corrcoef, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from transformers import WhisperProcessor, WhisperForAudioClassification

###############################################################################
## SET PARAMETERS

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
SAMPLING_RATE = 16000
datasets = ['EMOVO', 'Multilinguis']
MAX_DURATION = 30
max_length = MAX_DURATION * SAMPLING_RATE
label_mapping = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Joy': 3, 'Neutral': 4, 'Sadness': 5, 'Surprise': 6}
BATCH_SIZE = 32  # Define batch size

# Create folder for results
os.makedirs("analysis", exist_ok=True)
output_csv_path = os.path.join('analysis', "performance.csv")

# Parsing command-line arguments
parser = argparse.ArgumentParser(description="Script for testing models performance")
parser.add_argument('mode', type=str, help="'testing' : test all the models;\n- 'production' : test production model;")
args = parser.parse_args()

MODE = args.mode
###############################################################################
## METHODS

# Method to make predictions using the ONNX model in batch
def predict_emotion_batch(audio_paths, ort_session, processor):
    """Predict emotions using the optimized ONNX model in batch."""
    
    # Load all audio in batch
    audio_arrays = []
    for audio_path in audio_paths:
        audio_array, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)), mode="constant") if len(audio_array) < max_length else audio_array[:max_length]
        audio_arrays.append(audio_array)
    
    # Prepare inputs for the ONNX model
    inputs = processor(audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    input_values = inputs['input_features'].cpu().numpy()

    # Prepare input for the ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: input_values}

    # Get predictions from the ONNX model for all inputs
    logits = ort_session.run(None, ort_inputs)[0]
    probabilities = F.softmax(torch.tensor(logits), dim=-1).numpy()
    
    # Get prediction for each file
    predictions = []
    for prob in probabilities:
        predicted_class_id = np.argmax(prob)
        predictions.append((label_mapping[predicted_class_id], prob[predicted_class_id] * 100))
    
    return predictions

# Method to make predictions using the raw model
def predict_emotion(audio_path, model, processor):
    """Predicts emotions using the Transformer model."""
    
    audio_array, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
    audio_array = np.pad(audio_array, (0, max_length - len(audio_array)), mode="constant") if len(audio_array) < max_length else audio_array[:max_length]
    
    inputs = processor(audio_array, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move tensors to GPU

    with torch.no_grad():  # Disable gradient calculation to save memory
        logits = model(**inputs).logits
    
    probabilities = F.softmax(logits, dim=-1).cpu().numpy()[0]
    predicted_class_id = np.argmax(probabilities)
    
    return label_mapping[predicted_class_id], probabilities[predicted_class_id] * 100


###############################################################################
## LOAD MODEL

if MODE == 'production':
    # Load the optimized ONNX model
    onnx_model_path = "production_model/ser_model_consolidated.onnx"
    ort_session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])

    # Load the processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

###############################################################################
## START ANALYSIS

# Analysis for each dataset
if not os.path.exists(output_csv_path):
    analysis_df = pd.DataFrame(columns=["Model", "EMOVO accuracy", "RAVDESS accuracy", "CaFE accuracy", "CREMA-D accuracy", "emoDB accuracy", "avg"])
    analysis_df.to_csv(output_csv_path, index=False)
else:
    analysis_df = pd.read_csv(output_csv_path)

    
if MODE == 'testing':
    num_iterations = 1
else:
    num_iterations = 15

for i in range(num_iterations):

    if MODE == 'testing':
            MODEL_NAME = f"whisper/output/MODEL_{i}/ser_finetuned_model"
            model = WhisperForAudioClassification.from_pretrained(MODEL_NAME).to(device)
            processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    dataset_map = {}

    for dataset in datasets:
        audio_folder = f"input/datasets/{dataset}/audio"
        audio_files = os.listdir(audio_folder)
        results = []

        if MODE == 'testing':

            for filename in tqdm(audio_files, desc=f"Processing {dataset} with MODEL_{i}"):
                file_path = os.path.join(audio_folder, filename)
                prediction, probability = predict_emotion(file_path, model, processor)
                results.append({"file_name": filename, "emotion_prediction": prediction, "probability": round(probability, 2)})

        else:
            for batch_start in tqdm(range(0, len(audio_files), BATCH_SIZE), desc=f"Processing {dataset} with production model"):
                batch_files = audio_files[batch_start:batch_start + BATCH_SIZE]
                batch_paths = [os.path.join(audio_folder, file) for file in batch_files]
                
                # Predictions in batch
                predictions = predict_emotion_batch(batch_paths, ort_session, processor)
                
                # Add results to DataFrame
                for idx, (prediction, probability) in enumerate(predictions):
                    results.append({"file_name": batch_files[idx], "emotion_prediction": prediction, "probability": round(probability, 2)})

        results_df = pd.DataFrame(results)

        if MODE == 'testing':
            result_csv_filename = os.path.join("analysis", f"MODEL_{i}")
        else:
            result_csv_filename = os.path.join("analysis", 'production_model')

        os.makedirs(result_csv_filename, exist_ok=True)
        results_df.to_csv(f"{result_csv_filename}/{dataset}_predictions.csv", index=False)

        # Load the dataset containing the ground truth labels
        evaluations_file_path = f"input/datasets/{dataset}/metadata/evaluations.csv"
        evaluations_df = pd.read_csv(evaluations_file_path, dtype=str).filter(items=['emotion_recognized', 'file_name'])

        # Filter emotions based on the dataset
        emotions = ["Anger", "Disgust", "Joy", "Sadness", "Fear", 'Neutral', 'Surprise'] if dataset not in ('emoDB', 'CREMA-D') else ["Anger", "Disgust", "Joy", "Sadness", "Fear", 'Neutral']
        evaluations_df = evaluations_df[evaluations_df["emotion_recognized"].isin(emotions)]
        results_df = results_df[results_df["emotion_prediction"].isin(emotions)]

        merged_df = pd.merge(evaluations_df, results_df, on="file_name", how="inner").drop_duplicates().dropna()

        if MODE == 'testing':
            output_dir = os.path.join("analysis", f"MODEL_{i}", f"{dataset}_analysis")
        else:
            output_dir = os.path.join("analysis", 'production_model', f"{dataset}_analysis")

        os.makedirs(output_dir, exist_ok=True)
        
        merged_df.to_csv(os.path.join(output_dir, f"Merged_Data_{dataset}.csv"), index=False)
        conf_matrix_file = os.path.join(output_dir, "confusion_matrix.png")
        probabilities_file = os.path.join(output_dir, "prediction_probabilities.png")

        results_file = os.path.join(output_dir, f"Analysis_Report_{dataset}.txt")
        with open(results_file, "w") as f:
            f.write("Basic summary statistics:\n")
            f.write(str(merged_df.describe()) + "\n\n")
            
            f.write("Emotion distribution in 'emotion_recognized' (Ground Truth):\n")
            f.write(str(merged_df['emotion_recognized'].value_counts()) + "\n\n")
            
            f.write("Emotion distribution in 'emotion_prediction' (Model Predictions):\n")
            f.write(str(merged_df['emotion_prediction'].value_counts()) + "\n\n")
            
            Macro_F1 = f1_score(merged_df['emotion_recognized'], merged_df['emotion_prediction'], average="macro")
            f.write(f"Macro F1 Score: {Macro_F1:.4f}\n")

            WA = accuracy_score(merged_df['emotion_recognized'], merged_df['emotion_prediction'])
            f.write(f"WA (Weighted Accuracy): {WA:.4f}\n")

            conf_matrix = confusion_matrix(merged_df['emotion_recognized'], merged_df['emotion_prediction'])
            acc_per_class = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
            acc_per_class = acc_per_class[~np.isnan(acc_per_class)]
            UA = np.mean(acc_per_class)

            f.write(f"Unweighted Accuracy (UA): {UA:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(conf_matrix) + "\n\n")
            
            accuracy = accuracy_score(merged_df['emotion_recognized'], merged_df['emotion_prediction'])
            f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n\n")

            uar = recall_score(merged_df['emotion_recognized'], merged_df['emotion_prediction'], average="macro")
            f.write(f"Unweighted Average Recall (UA): {uar:.4f}\n\n")

            war = recall_score(merged_df['emotion_recognized'], merged_df['emotion_prediction'], average="weighted")
            f.write(f"Weighted Average Recall (WA): {war:.4f}\n\n")

            mcc_score = matthews_corrcoef(merged_df['emotion_recognized'], merged_df['emotion_prediction'])
            f.write(f"Matthews Correlation Coefficient (MCC): {mcc_score:.4f}\n\n")

            class_report = classification_report(merged_df['emotion_recognized'], merged_df['emotion_prediction'], target_names=emotions)
            f.write("Classification Report:\n")
            f.write(class_report + "\n\n")
            
            correlation = merged_df['emotion_recognized'].astype('category').cat.codes.corr(merged_df['emotion_prediction'].astype('category').cat.codes)
            f.write(f"Correlation between Ground Truth and Predictions: {correlation}\n\n")

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=emotions, yticklabels=emotions)
        plt.xlabel('Predicted Emotion')
        plt.ylabel('Actual Emotion')
        plt.title('Confusion Matrix - Model vs Ground Truth')
        plt.savefig(conf_matrix_file)
        plt.close()

        print(f"Analysis saved in {results_file}\nPlots saved in {output_dir}")
        print("#"*70)