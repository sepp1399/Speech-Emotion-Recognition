"""
TITLE: "Model converter for batch and fast inference"
AUTHORS: Giuseppe Lentini
LAST UPDATE: 2025-04-02
PYTHON VERSION: 3.11.11
"""

###############################################################################
## IMPORTING LIBRARIES
import torch
from transformers import WhisperProcessor, WhisperForAudioClassification
import os
import onnx
from onnx.external_data_helper import convert_model_to_external_data

###############################################################################
## PATHS
model_path = "./whisper/output/MODEL_XX/ser_finetuned_model"
onnx_model_path = "/whisper/temp/ser_model_optimized.onnx"
tensorrt_model_path = "/whisper/temp/ser_model_optimized.trt"
os.makedirs("production_model", exist_ok=True)

os.makedirs("/whisper/temp", exist_ok=True)
###############################################################################
## 1. LOAD MODEL & PROCESSOR
model = WhisperForAudioClassification.from_pretrained(model_path).to("cuda").eval()
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

###############################################################################
## 2. GENERATE DUMMY INPUT
batch_size = 1
dummy_waveforms = [torch.randn(480000) for _ in range(batch_size)]  # ~30s audio @16kHz
dummy_np = [wave.cpu().numpy() for wave in dummy_waveforms]

# Feature extraction
features = processor.feature_extractor(dummy_np, sampling_rate=16000, return_tensors="pt", padding=True)
input_features = features["input_features"].to("cuda")

###############################################################################
## 3. EXPORT TO ONNX
torch.onnx.export(
    model,
    (input_features,),
    onnx_model_path,
    input_names=["input_features"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={
        "input_features": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
)
print(f"✅ ONNX model exported to: {onnx_model_path}")

model = onnx.load(onnx_model_path)

convert_model_to_external_data(
    model,
    all_tensors_to_one_file=True,
    location="model.data",
    size_threshold=0,
    convert_attribute=False
)

onnx.save_model(
    model,
    "production_model/ser_model_consolidated.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="model.data",
    size_threshold=0
)

print("✅ Consolidated ONNX saved as 'ser_model_consolidated.onnx'")