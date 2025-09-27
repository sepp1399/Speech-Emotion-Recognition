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
# import required Python libraries
import io
import sys
import json
import os
import numpy as np
import pandas as pd
from multiprocessing import cpu_count  
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import mixed_precision
import time
import seaborn as sns
import matplotlib.pyplot as plt
import tf2onnx
import onnx
import torch
from numba import cuda 
import gc
import openvino.runtime as ov

###############################################################################
## 2. PARAMETERS TO BE SET!!!

# set the correct pathways/folders
BASE_DIR_INPUT = ('.')
BASE_DIR_OUTPUT = BASE_DIR_INPUT

# set input/output file names
model_number = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         'config/model_number.csv']), header = None, 
                                     dtype = 'str').iloc[0].values[0]
input_file_name_05 = ('config/run_modality.csv')
input_file_name_06 = ('config/env.csv')
output_file_name_28 = (f'output/{model_number}/')

# environment:
# 'local' or 'cloud' (in order to skip gpus usage if in local)
env = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         input_file_name_06]), header = None, 
                                     dtype = 'str')

# setup logging
LOG_DIR = os.path.join("output", model_number, "log")
LOG_FILE = os.path.join(LOG_DIR, "03_training_log.txt")

# log function
def log(message, do_print=True):
    """Writes a message to both the console and the log file."""
    
    if do_print:
        print(message)

    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

###############################################################################
## 3. USING GPU

# Enable dynamic GPU memory allocation
if env.iloc[0, 0] == 'cloud':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    #select specific GPU
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

    #improve gpu performances
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    #Use XLA (Accelerated Linear Algebra)
    tf.config.optimizer.set_jit(True)
else:

    # Create an instance of the OpenVINO core
    core = ov.Core()

    # Check available devices (CPU, GPU, MYRIAD, etc.)
    available_devices = core.available_devices
    print("Available devices:", available_devices)

    # Set the best available device (GPU preferred, otherwise CPU)
    device = "GPU" if "GPU" in available_devices else "CPU"
    print(f"Using OpenVINO device: {device}")

    # Optimize OpenVINO for Intel GPU
    if device == "GPU":
        core.set_property("GPU", {
            "CACHE_DIR": "./ov_cache",  # Enable caching
            "NUM_STREAMS": "AUTO",  # Max parallelism
            "INFERENCE_PRECISION_HINT": "f16",  # Use FP16 for better performance
            "PERFORMANCE_HINT": "THROUGHPUT"
        })
    elif device == "CPU":
        core.set_property("CPU", {
            "CPU_THREADS_NUM": str(os.cpu_count()),  # Use all CPU cores
            "CACHE_DIR": "./ov_cache",  # Enable caching
            "INFERENCE_PRECISION_HINT": "f32",  # FP32 for best CPU accuracy
            "CPU_THROUGHPUT_STREAMS": "AUTO"  # Auto parallelization
        })

    # Enable OneDNN optimizations (for TensorFlow on Intel CPU)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

    # Enable XLA (Accelerated Linear Algebra) if using TensorFlow
    tf.config.optimizer.set_jit(True)

    print(f"Optimized OpenVINO backend set to: {device}")


###############################################################################
## 4. LOADING DATA-SET 

# start clocking time
start_time = time.time()

 
# run_modality, either: 
# - 'testing' : without grid-search => quick; 
# - 'production' : with grid-search => slow;
run_modality = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         input_file_name_05]), header = None, 
                                     dtype = 'str')

# load already pre-processed training-set
train_set = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                           ('{}/data/train_set.csv'.format(
                                           output_file_name_28))]), header = 0)

# load preprocessed data used previously for hyperparameter tuning
additional_data = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                           ('{}/data/validation1_set.csv'.format(
                                           output_file_name_28))]), header = 0)

# # load already pre-processed validation-set
val_set = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                           ('{}/data/validation2_set.csv'.format(
                                           output_file_name_28))]), header = 0)

################################################################################
## 5. PRE-PROCESSING

log("=== TRAINING START ===")

# # merge train set and additional data in order to increase data for training (if few data)
# train_set = pd.concat([train_set, additional_data], ignore_index=True)

log(f"Train set size: {len(train_set)}", False)
log(f"Validation set size: {len(val_set)}", False)
log("=================================\n", False)

# Log distribution of classes for training and validation sets
train_dist = train_set['emotions'].value_counts()
train_dist_perc = train_set['emotions'].value_counts(normalize=True) * 100

val_dist = val_set['emotions'].value_counts()
val_dist_perc = val_set['emotions'].value_counts(normalize=True) * 100

log("=== Dataset Class Distributions ===", False)

train_summary = "\n".join([f"{cls}: {count} ({perc:.2f}%)" 
                           for cls, count, perc in zip(train_dist.index, train_dist.values, train_dist_perc.values)])
val_summary = "\n".join([f"{cls}: {count} ({perc:.2f}%)" 
                         for cls, count, perc in zip(val_dist.index, val_dist.values, val_dist_perc.values)])

log(f"Train set:\n{train_summary}\n", False)
log(f"Validation set:\n{val_summary}\n", False)


# initialize the encoding of labels/strings into integers
le = LabelEncoder()

# label encoding
train_set.loc[:, 'emotions'] = le.fit_transform(train_set.loc[:, 'emotions'])
val_set.loc[:, 'emotions'] = le.transform(val_set.loc[:, 'emotions'])

# split dependent (y_train) from independent (X_train) variables
y_train = train_set.pop("emotions")
y_val = val_set.pop("emotions")

# convert to Numpy
y_train_to_numpy = y_train.to_numpy(dtype="int")
y_val_to_numpy = y_val.to_numpy(dtype="int") 

# pop file_name column
file_name_series_train = train_set.pop("file_name")
file_name_series_val = val_set.pop("file_name") 

# pop subject_serial_number
subject_serial_number_train = train_set.pop("subject_serial_number")
subject_serial_number_val = val_set.pop("subject_serial_number") 

# deep copy
X_train = train_set.copy()
X_val = val_set.copy() 

# convert to Numpy
X_train_to_numpy = X_train.to_numpy()
X_val_to_numpy = X_val.to_numpy() 

# Converting Pandas DataFrame to TensorFlow dataset
train_set_tf = tf.data.Dataset.from_tensor_slices((tf.expand_dims(X_train_to_numpy, axis=2), y_train_to_numpy))
val_set_tf = tf.data.Dataset.from_tensor_slices((tf.expand_dims(X_val_to_numpy, axis=2), y_val_to_numpy))

# set fitting process according to either 'testing' or 'production'
if run_modality.iloc[0, 0] == 'testing': 
    num_epochs=6
elif run_modality.iloc[0, 0] == 'production':
    num_epochs=1000


# set batch size, training size, steps_per_epoch for train-set
training_size=X_train.shape[0]
batch_size=64
patience=64 # for early stopping
# steps_per_epoch=np.ceil(training_size/batch_size)
ds_train = train_set_tf.cache().shuffle(buffer_size = training_size).batch(batch_size, drop_remainder = True).prefetch(buffer_size = training_size)
# ds_train = train_set.cache().shuffle(buffer_size = training_size).repeat().batch(batch_size, drop_remainder = True).prefetch(buffer_size = training_size) # not working without specificing "steps_per_epoch" explicitly
ds_val = val_set_tf.cache().batch(batch_size, drop_remainder=False).prefetch(buffer_size=training_size)

##############################################################################
## 6. DEEP LEARNING FINAL MODEL: 
# load the best hyperparameters got from the training step    
param_grid = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                            ('{}analysis_best_scores.csv'.format(
                                            output_file_name_28))]), header = 0)

# sub-set best_score data-frame by the units size for each level of layers
param_grid_units_tmp_gr_01 = param_grid.loc[0, param_grid.columns.str.contains('filters_for_layer_gr_01')].astype('int') 
# param_grid_units_tmp_gr_02 = param_grid.loc[0, param_grid.columns.str.contains('units_for_layer_gr_02')].astype('int') 

# re-set index
param_grid_units_gr_01 = param_grid_units_tmp_gr_01.reset_index(drop = False, inplace = False)
# param_grid_units_gr_02 = param_grid_units_tmp_gr_02.reset_index(drop = False, inplace = False)

# rename columns
param_grid_units_gr_01.rename(columns = {'index': 'layer_name', 
                                   0: 'n_filters'}, inplace = True)
# param_grid_units_gr_02.rename(columns = {'index': 'layer_name', 
#                                    0: 'n_filters'}, inplace = True)


# function for initializing, building and compiling the Keras' model
def build_model(param_grid_units_gr_01, param_grid):
    
    """Function used by Keras-tuner for doing hyperparameter tuning.
      Args:
        param_grid_units_gr_01: set of hyperparameters for layer 1.
        param_grid: general set of hyperparameters.   
        
      Returns:
        model: compiled Keras model. 
    """
    
    # initialize model
    model = tf.keras.models.Sequential()
    
    # Layer 01
    model.add(tf.keras.layers.Conv1D(filters=list(param_grid.loc[:, 'filters_for_layer_gr_01'].astype("int"))[0],
                                    kernel_size=list(param_grid.loc[:, 'kernel_gr_01'].astype("int"))[0], 
                                    padding='same', 
                                    strides=1,                                 
                                    input_shape=(X_train.columns.shape[0], 1),
                                    name='layer_gr_01'))
    model.add(tf.keras.layers.BatchNormalization())  
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(rate=list(np.around(param_grid.loc[:, 'dropout_gr_01'], decimals=4))[0], 
                                      name='dropout_gr_01')) 
    
    # Layer 02
    model.add(tf.keras.layers.Conv1D(filters=128,
                                kernel_size=list(param_grid.loc[:, 'kernel_gr_02'].astype("int"))[0], 
                                padding='same', 
                                strides=1,
                                name='layer_gr_02'))
    model.add(tf.keras.layers.BatchNormalization())  
    model.add(tf.keras.layers.Activation('relu'))   
    model.add(tf.keras.layers.Dropout(rate=list(np.around(param_grid.loc[:, 'dropout_gr_02'], decimals=4))[0], 
                                      name='dropout_gr_02')) 

    # Max Pooling
    model.add(tf.keras.layers.MaxPooling1D(pool_size=list(param_grid.loc[:, 'pool_gr_01'].astype("int"))[0], 
                                           name='pool_gr_01'))
    
    # Layer 03
    model.add(tf.keras.layers.Conv1D(filters=128,
                                kernel_size=list(param_grid.loc[:, 'kernel_gr_03'].astype("int"))[0], 
                                padding='same', 
                                strides=1,
                                name='layer_gr_03'))
    model.add(tf.keras.layers.BatchNormalization())  
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(rate=list(np.around(param_grid.loc[:, 'dropout_gr_03'], decimals=4))[0], 
                                      name='dropout_gr_03')) 
    
    # Layer 04
    model.add(tf.keras.layers.Conv1D(filters=128,
                                kernel_size=list(param_grid.loc[:, 'kernel_gr_04'].astype("int"))[0], 
                                padding='same', 
                                strides=1,
                                name='layer_gr_04'))
    model.add(tf.keras.layers.BatchNormalization())  
    model.add(tf.keras.layers.Activation('relu'))  
    model.add(tf.keras.layers.Dropout(rate=list(np.around(param_grid.loc[:, 'dropout_gr_04'], decimals=4))[0], 
                                      name='dropout_gr_04'))  
    
    # Flatten
    model.add(tf.keras.layers.Flatten())
    
    # Extra Dense Layer
    model.add(tf.keras.layers.Dense(units=list(param_grid.loc[:, 'extra_dense_layer'].astype("int"))[0], 
                                    name='extra_dense_layer'))
    model.add(tf.keras.layers.BatchNormalization())  
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(rate=list(np.around(param_grid.loc[:, 'dropout_gr_05'], decimals=4))[0], 
                                      name='dropout_gr_05')) 
    
    # Output Layer
    model.add(tf.keras.layers.Dense(units=len(pd.unique(y_train)), 
                                    activation='softmax', 
                                    name='output'))
       
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=param_grid.loc[0, 'learning_rate'],
            weight_decay=param_grid.loc[0, 'weight_decay']
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    
    return model

# build model Keras' model
model = build_model(param_grid_units_gr_01,
                    # param_grid_units_gr_02,
                    param_grid)

# model fitting
fitting_results=model.fit(ds_train, 
                    epochs=num_epochs,
                    validation_data=ds_val,
                    callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy', patience=patience)],
                    )

# log model summary
buffer = io.StringIO()
sys.stdout = buffer
model.summary()
sys.stdout = sys.__stdout__
model_summary = buffer.getvalue()
log(model_summary, False)

metrics = {
    "best_val_accuracy": max(fitting_results.history['val_accuracy']),
    "best_epoch": fitting_results.history['val_accuracy'].index(max(fitting_results.history['val_accuracy'])) + 1,
    "final_train_accuracy": fitting_results.history['accuracy'][-1],    
    "final_train_loss": fitting_results.history['loss'][-1],
    "final_val_accuracy": fitting_results.history['val_accuracy'][-1],
    "final_val_loss": fitting_results.history['val_loss'][-1]
}

metrics_str = json.dumps(metrics, indent=4)

log(metrics_str, False)
log("=================================\n")

# Create 'model' folder
model_dir = os.path.join(BASE_DIR_OUTPUT, 'output', model_number, 'saved_model')
os.makedirs(model_dir, exist_ok=True)

# save Keras' model (=> weights)
model.save(os.path.join(model_dir, 'best_hyper_model.hdf5'))

# save model according to TF's convention
tf.keras.models.save_model(model, os.path.join(model_dir, 'best_hyper_model.keras'))

# imput shape
input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input_data")]

# save model in onnx format
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, os.path.join(model_dir,"best_hyper_model.onnx"))

################################################################################
## 6. PLOTTING DATA

# create DataFrame
performance_train_loss =  pd.DataFrame(data=(fitting_results.history['loss'][11:]), 
                                     columns=['loss'])

performance_train_accuracy =  pd.DataFrame(data=(fitting_results.history['accuracy'][11:]), 
                                      columns=['accuracy'])

performance_val_loss =  pd.DataFrame(data=(fitting_results.history['val_loss'][11:]), 
                                     columns=['val_loss'])
                                     
performance_val_accuracy =  pd.DataFrame(data=(fitting_results.history['val_accuracy'][11:]), 
                                      columns=['val_accuracy'])

# concatenate masks
performance_concat = pd.concat([performance_train_loss,
                                performance_train_accuracy], axis = 1)

# plot data 
sns.lineplot(data=performance_concat)
plt.title('model loss/accuracy')
plt.ylabel('loss/accuracy')
plt.xlabel('epoch')

# save plot as .pdf file
plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, ('{}loss_profile.pdf'.format(
                output_file_name_28))]))

plt.close()

# concatenate masks
performance_concat2 = pd.concat([performance_val_loss, performance_val_accuracy], axis = 1)

# plot data 
sns.lineplot(data=performance_concat2)
plt.title('model val loss/accuracy')
plt.ylabel('val loss/accuracy')
plt.xlabel('epoch')

# save plot as .pdf file
plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, ('{}val_loss_profile.pdf'.format(
                output_file_name_28))]))

plt.close()

# end time according to computer clock
end_time = time.time()

# calculate total execution time
total_execution_time = pd.Series(np.round((end_time - start_time), 2)).rename('total_training_runtime_seconds')

# shows run-time's timestamps + total execution time
log("=== TRAINING FINISHED ===\n")

log('start time (unix timestamp):{}'.format(start_time))
log('end time (unix timestamp):{}'.format(end_time))
log('total execution time (seconds):{}'.format(total_execution_time.iloc[0]))
log("=================================\n")

# Clear TensorFlow session
tf.keras.backend.clear_session()

# Release GPU memory in PyTorch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Release GPU memory in Numba
    cuda.select_device(0)
    cuda.close()

# Force garbage collection to free CPU memory
gc.collect()
