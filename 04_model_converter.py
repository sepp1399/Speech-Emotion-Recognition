"""
TITLE: "Model converter for batch and fast inference"
AUTHORS: Giuseppe Lentini
LAST UPDATE: 2025-04-02
PYTHON VERSION: 3.11.11
"""

###############################################################################
## IMPORTING LIBRARIES
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import os
import onnx
import onnxruntime as ort
from onnxruntime import GraphOptimizationLevel, SessionOptions
import tensorrt as trt
import sys

###############################################################################
## PATHS
model_path = "./best_model/ser_finetuned_model"
onnx_model_path = "production_model/ser_model_optimized.onnx"
tensorrt_model_path = "production_model/ser_model_optimized.trt"

os.makedirs("production_model", exist_ok=True)

###############################################################################
## METHODS
def convert_onnx_to_tensorrt(onnx_model_path, trt_engine_path, fp16_mode=False):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4 GB

        if fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)

        # load ONNX model
        with open(onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('Failed to parse ONNX model')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Create profile optimizer
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)

        # manage batch size
        profile.set_shape(input_tensor.name, min=(1, 64000), opt=(8, 64000), max=(16, 64000))
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build TensorRT engine")
            return None

        with open(trt_engine_path, "wb") as f:
            f.write(serialized_engine)

        print(f"TensorRT engine saved to {trt_engine_path}")

###############################################################################
## CONVERT

# 1. Load the model and processor
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to("cuda")
model.eval()
processor = Wav2Vec2Processor.from_pretrained(model_path)

# 2. Create a dummy audio batch to simulate input
batch_size = 32  # Batch size
dummy_audio = torch.randn(batch_size, 64000, dtype=torch.float32, device="cuda")  # 4 seconds at 16kHz (64000 samples)
# Here, .squeeze() is not necessary since the shape is already correct

# 3. Process the dummy audio batch
inputs = processor(dummy_audio, return_tensors="pt", padding=True, sampling_rate=16000)

# 4. Move the inputs to the GPU
inputs = {key: val.to("cuda", dtype=torch.float32) for key, val in inputs.items()}  # Changed to float32
inputs['input_values'] = inputs['input_values'].squeeze()  # Removes the extra dimension if present

# 5. Export the model to ONNX format with dynamic batch size
torch.onnx.export(model, 
                  inputs['input_values'], 
                  onnx_model_path, 
                  input_names=["input_values"], 
                  output_names=["logits"], 
                  opset_version=17,
                  dynamic_axes={"input_values" : {0 : "batch_size"}, "logits" : {0 : "batch_size"}})

print(f"ONNX model exported as: {onnx_model_path}")

# Convert the ONNX model to TensorRT
convert_onnx_to_tensorrt(onnx_model_path, tensorrt_model_path, fp16_mode=True)
