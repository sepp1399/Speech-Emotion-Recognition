# Talking About - Speech Emotion Recognition

## 📌 Description
This project leverages a **Transformer-based model** for **Speech Emotion Recognition**. The pipeline includes:
- **Data Preprocessing**
- **Model Training**
- **Evaluation and Inference**

## 📂 Project Structure

### **1. 01_transformer_training.py** (Model Training & Evaluation)
This script handles the end-to-end process of training and evaluating a **Transformer-based model** for speech emotion recognition.
- Loads and preprocesses the dataset (including feature extraction).
- Uses a **pretrained transformer** for feature representation.
- Fine-tunes the model on emotion recognition data.
- Evaluates performance on the test set.
- Saves the trained model and results.

### **2. 02_transformer_testing.py** (Model Testing & Batch Inference)

This script is designed for evaluating a trained Speech Emotion Recognition (SER) model on unseen data. It supports both standard PyTorch inference and optimized ONNX-based batch inference.

#### **Features**
- Loads trained models from `output/` (for PyTorch testing) or `production_model/` (for ONNX inference).
- Utilizes `argparse` to allow flexible command-line execution.
- Preprocesses test audio, including feature extraction, normalization, and padding.
- Supports **both single-instance and batch inference** for optimized performance.
- Runs predictions using either **PyTorch** or **ONNX-runtime** for faster execution.
- Computes and saves key performance metrics (accuracy, F1-score, UAR, WAR, MCC).
- Outputs detailed **confusion matrices** and **classification reports** to `analysis/`.

#### **Usage**
```bash
python 02_transformer_testing.py testing/production
```

### **3. 03_model_converter.py** (Model Conversion for Deployment)
This script converts a trained PyTorch model into the **ONNX format and tensorRT** to improve inference speed and compatibility.
- Loads a trained Transformer model from `output/`.
- Converts the model to ONNX and tensorRT format for optimized inference.
- Enables deployment on platforms supporting ONNX runtime.

## ⚙️ Requirements
This project has been tested with **Python 3.11.11**. Install dependencies using:
```sh
pip install --upgrade -r requirements.txt
```

Dependencies include:
- `torch`, `torchvision`, `torchaudio`
- `pandas`, `numpy`, `scikit-learn`
- `transformers`, `soundfile`, `evaluate`, `datasets`

For more details, check `requirements.txt`.

## 🚀 Execution
To train the model, run:
```sh
python 01_transformer_training.py
```

## 📊 Output
- **Trained Model**: Saved in the `output/` directory.
- **Evaluation Results**: Includes accuracy, confusion matrix, and other metrics.
- **Predictions**: Can be used for further analysis.

---
✉️ **Authors**: Giuseppe Lentini
