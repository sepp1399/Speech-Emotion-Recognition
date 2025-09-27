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
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
from tensorflow.keras import mixed_precision
import keras_tuner as kt
import time
from numba import cuda 
import gc
import openvino.runtime as ov
import torch

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
LOG_FILE = os.path.join(LOG_DIR, "02_hyperparameters_tuning_log.txt")

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

# load already pre-processed data-set
whole_data_set_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                           ('processed_data/preprocessed_set.csv')]), header = 0)

################################################################################
## 5. PRE-PROCESSING

def balance_score(class_counts):
    """ Computes a balance score based on class distribution variance. 
        Lower variance means better balance. """
    counts = np.array(list(class_counts.values()))
    percentages = counts / counts.sum()  # Normalize as percentages
    variance = np.var(percentages)  # Lower variance is better
    return variance

log("=== HYPERPARAMETERS TUNING START ===")
random_state = np.random.randint(0, 10000)

# if not already preproccesed
if not os.path.exists(output_file_name_28 + "data/train_set.csv"):

    # Generate a new random state in order to have different splits during various iterations
    log(f'random state: {random_state}')

    # convert "object" to strings
    whole_data_set = whole_data_set_tmp.astype({"emotions": "string",
                        "file_name": "string",
                        "subject_serial_number": "string"})


    # Set validation size (25% validation, 75% training)
    validation_size = 0.25
    n_splits = round(1 / validation_size)  # round(1 / 0.3) = 4

    # Split train-validation usando StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    validation_idxs, train_idxs = next(sgkf.split(whole_data_set, 
                                                whole_data_set["emotions"], 
                                                whole_data_set["subject_serial_number"]))

    # train set must be bigger than validation
    if len(validation_idxs) > len(train_idxs):
        validation_idxs, train_idxs = train_idxs, validation_idxs

    # extract train_set    
    train_set_tmp = whole_data_set.iloc[train_idxs, :]

    # extract test_set    
    validation_set_tmp_tmp = whole_data_set.iloc[validation_idxs, :]

    # Set validation size (60% validation1, 40% validation2)
    validation_size = 0.40
    n_splits = round(1 / validation_size)  # round(1 / 0.4) = 3

    # Split validation into two subsets (one for hyperparameter tuning, one for final training)
    sgkf_val = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    validation1_idxs, validation2_idxs = next(sgkf_val.split(validation_set_tmp_tmp, 
                                                            validation_set_tmp_tmp["emotions"], 
                                                            validation_set_tmp_tmp["subject_serial_number"]))

    # double check validation
    if len(validation2_idxs) > len(validation1_idxs):
        validation1_idxs, validation2_idxs = validation2_idxs, validation1_idxs

    # Extract the subsets
    validation1_set = validation_set_tmp_tmp.iloc[validation1_idxs, :]
    validation2_set = validation_set_tmp_tmp.iloc[validation2_idxs, :]

    # Split validation into two subsets (one for hyperparameter tuning, one for final training)
    sgkf_val = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    validation1_final_idxs, test_idxs = next(sgkf_val.split(validation1_set, 
                                                            validation1_set["emotions"], 
                                                            validation1_set["subject_serial_number"]))

    # double check validation
    if len(test_idxs) > len(validation1_idxs):
        validation1_final_idxs, test_idxs = test_idxs, validation1_final_idxs

    # Extract the subsets
    validation1_set_final = validation1_set.iloc[validation1_final_idxs, :]
    test_set = validation1_set.iloc[test_idxs, :]

    log(f"Train set size: {len(train_set_tmp)}")
    log(f"Validation1 set size: {len(validation1_set)}")
    log(f"validation2 set size: {len(validation2_set)}")
    log(f"Test set size: {len(test_set)}")

    # Save the cleaned datasets
    train_set_tmp.to_csv(output_file_name_28 + "data/train_set.csv", index=False)
    validation1_set.to_csv(output_file_name_28 + "data/validation1_set.csv", index=False)
    validation2_set.to_csv(output_file_name_28 + "data/validation2_set.csv", index=False)
    test_set.to_csv(output_file_name_28 + "data/test_set.csv", index=False)

# Compute class distributions
    val1_counts = validation1_set["emotions"].value_counts().to_dict()
    val2_counts = validation2_set["emotions"].value_counts().to_dict()

    # Compute balance scores for both subsets
    score_val1 = balance_score(val1_counts)
    score_val2 = balance_score(val2_counts)

    # If validation2_set is more balanced, swap the subsets
    if score_val2 < score_val1:
        validation1_set, validation2_set = validation2_set, validation1_set
        log("Swapped validation1 and validation2 for better balance.")

    # Print final balance scores
    log(f"Final Validation1 balance score: {balance_score(validation1_set['emotions'].value_counts().to_dict())}")
    log(f"Final Validation2 balance score: {balance_score(validation2_set['emotions'].value_counts().to_dict())}")


    # Log distribution of classes for training and validation sets
    log("=== Dataset Class Distributions ===", False)

    train_dist = train_set_tmp['emotions'].value_counts()
    train_dist_perc = train_set_tmp['emotions'].value_counts(normalize=True) * 100

    val1_dist = validation1_set['emotions'].value_counts()
    val1_dist_perc = validation1_set['emotions'].value_counts(normalize=True) * 100

    val2_dist = validation2_set['emotions'].value_counts()
    val2_dist_perc = validation2_set['emotions'].value_counts(normalize=True) * 100

    test_dist = test_set['emotions'].value_counts()
    test_dist_perc = test_set['emotions'].value_counts(normalize=True) * 100

    train_summary = "\n".join([f"{cls}: {count} ({perc:.2f}%)" for cls, count, perc in zip(train_dist.index, train_dist.values, train_dist_perc.values)])
    val1_summary = "\n".join([f"{cls}: {count} ({perc:.2f}%)" for cls, count, perc in zip(val1_dist.index, val1_dist.values, val1_dist_perc.values)])
    val2_summary = "\n".join([f"{cls}: {count} ({perc:.2f}%)" for cls, count, perc in zip(val2_dist.index, val2_dist.values, val2_dist_perc.values)])
    test_summary = "\n".join([f"{cls}: {count} ({perc:.2f}%)" for cls, count, perc in zip(test_dist.index, test_dist.values, test_dist_perc.values)])

    log(f"Train set:\n{train_summary}", False)
    log(f"\nValidation1 set:\n{val1_summary}\n", False)
    log(f"\nValidation2 set:\n{val2_summary}\n", False)
    log(f"\ntest set:\n{test_summary}\n", False)

    # size of each set
    overall_count = pd.unique(whole_data_set.loc[:, "subject_serial_number"]).size
    train_set_count = pd.unique(train_set_tmp.loc[:, "subject_serial_number"]).size
    validation_set_count = pd.unique(validation_set_tmp_tmp.loc[:, "subject_serial_number"]).size

    # count single IDs
    log('total ID count: {}'.format(overall_count))
    log('train_set ID count: {}'.format(train_set_count))
    log('test_set ID count: {}'.format(validation_set_count))
    log('percentage train_set: {}'.format((train_set_count)/overall_count))

# Get unique actor IDs from each subset
    train_actors = set(train_set_tmp["subject_serial_number"])
    validation1_actors = set(validation1_set["subject_serial_number"])
    validation2_actors = set(validation2_set["subject_serial_number"])
    test_actors = set(test_set["subject_serial_number"])

    # Check for overlaps
    train_val1_overlap = train_actors & validation1_actors
    train_val2_overlap = train_actors & validation2_actors
    train_test_overlap = train_actors & test_actors
    val1_val2_overlap = validation1_actors & validation2_actors

    # Print results
    if not train_val1_overlap and not train_val2_overlap and not train_test_overlap and not val1_val2_overlap:
        log("No actors are shared between subsets.")
    else:
        log("WARNING: Some actors appear in multiple subsets!")
        if train_val1_overlap:
            log(f"Overlap between Train and Validation1: {train_val1_overlap}")
        if train_val2_overlap:
            log(f"Overlap between Train and Validation2: {train_val2_overlap}")
        if train_test_overlap:
            log(f"Overlap between Train and Test Set: {train_test_overlap}")
        if val1_val2_overlap:
            log(f"Overlap between Validation1 and Validation2: {val1_val2_overlap}")

    log("=================================\n")

# open sets
train_set_tmp = pd.read_csv(output_file_name_28 + "data/train_set.csv")
validation_set_tmp = pd.read_csv(output_file_name_28 + "data/validation1_set.csv")

# pop dependent variable
y_train = train_set_tmp.pop("emotions")

# pop file_name column
file_name_series_train = train_set_tmp.pop("file_name")

# pop subject_serial_number
subject_serial_number_train = train_set_tmp.pop("subject_serial_number")

# pop dependent variable
y_validation = validation_set_tmp.pop("emotions")

# pop file_name column
file_name_series_test = validation_set_tmp.pop("file_name")

# pop subject_serial_number
subject_serial_number_test = validation_set_tmp.pop("subject_serial_number")

# initialize label encoding
le = LabelEncoder()

# label encoding
y_train= le.fit_transform(y_train)
y_validation= le.fit_transform(y_validation)

# deep copy()
X_train = train_set_tmp.copy()
X_validation = validation_set_tmp.copy()
  
# converting Pandas DataFrame to TensorFlow data-sets
training_set = tf.data.Dataset.from_tensor_slices((tf.expand_dims(X_train, axis=2), y_train))
validation_set = tf.data.Dataset.from_tensor_slices((tf.expand_dims(X_validation, axis=2), y_validation))                                      

# set batch size, training size etc. for train-set + validation-set
training_size=X_train.shape[0]
validation_size=X_validation.shape[0]
batch_size=64
ds_train = training_set.cache().shuffle(buffer_size = training_size).batch(batch_size, drop_remainder = True).prefetch(buffer_size = training_size)
ds_validation = validation_set.cache().batch(batch_size, drop_remainder = True).prefetch(buffer_size = validation_size)


##############################################################################
## 6. HYPERPARAMETER TUNING BY RANDOMIZED GRID SEARCH (goal: find the optimal 
## hyperparameters)


# set hyperparameter to be search according to either 'testing' or 'production'
if run_modality.iloc[0, 0] == 'testing':
    min_value_filters_gr_01=129
    max_value_filters_gr_01=256
    step_filters_gr_01=16
    min_value_kernel_gr_01=3
    max_value_kernel_gr_01=7
    step_kernel_gr_01=1
    min_value_kernel_gr_02=3
    max_value_kernel_gr_02=7
    step_kernel_gr_02=1
    min_value_kernel_gr_03=3
    max_value_kernel_gr_03=7
    step_kernel_gr_03=1
    min_value_kernel_gr_04=3
    max_value_kernel_gr_04=7
    step_kernel_gr_04=1
    min_value_strides=1
    max_value_strides=3
    step_strides=1    
    values_l1=np.linspace(start = 0.001, 
                          stop = 0.9, 
                          num = 10)
    values_l2=np.linspace(start = 0.001, 
                          stop = 0.9, 
                          num = 10)
    min_value_dropout_gr_01=0.0
    max_value_dropout_gr_01=0.5
    step_dropout_gr_01=0.1
    min_value_dropout_gr_02=0.0
    max_value_dropout_gr_02=0.5
    step_dropout_gr_02=0.1
    min_value_dropout_gr_03=0.0
    max_value_dropout_gr_03=0.5
    step_dropout_gr_03=0.1
    min_value_dropout_gr_04=0.0
    max_value_dropout_gr_04=0.5
    step_dropout_gr_04=0.1
    min_value_dropout_gr_05=0.0
    max_value_dropout_gr_05=0.5
    step_dropout_gr_05=0.1
    min_value_pool=3
    max_value_pool=9
    step_pool=1
    min_value_extra=16
    max_value_extra=128
    step_extra=8
    values_learning_rate=np.linspace(start = 0.00001, stop = 0.1, num = 2) #linear
    max_trials=1 # Is it like the sample size sampled from the hyperparameter space?
    executions_per_trial=1 # Is it like K-folds?
    epochs=50
    patience=100
    
elif run_modality.iloc[0, 0] == 'production':
    min_value_filters_gr_01=128
    max_value_filters_gr_01=256
    step_filters_gr_01=16
    min_value_kernel_gr_01=3
    max_value_kernel_gr_01=7
    step_kernel_gr_01=1
    min_value_kernel_gr_02=3
    max_value_kernel_gr_02=7
    step_kernel_gr_02=1
    min_value_kernel_gr_03=3
    max_value_kernel_gr_03=7
    step_kernel_gr_03=1
    min_value_kernel_gr_04=3
    max_value_kernel_gr_04=7
    step_kernel_gr_04=1
    min_value_strides=1
    max_value_strides=3
    step_strides=1
    values_l1 = np.logspace(np.log10(0.0001), np.log10(0.1), num=10)
    values_l2 = np.logspace(np.log10(0.0001), np.log10(0.1), num=10)
    min_value_dropout_gr_01=0.0
    max_value_dropout_gr_01=0.5
    step_dropout_gr_01=0.1
    min_value_dropout_gr_02=0.0
    max_value_dropout_gr_02=0.5
    step_dropout_gr_02=0.1
    min_value_dropout_gr_03=0.0
    max_value_dropout_gr_03=0.5
    step_dropout_gr_03=0.1
    min_value_dropout_gr_04=0.0
    max_value_dropout_gr_04=0.5
    step_dropout_gr_04=0.1
    min_value_dropout_gr_05=0.0
    max_value_dropout_gr_05=0.5
    step_dropout_gr_05=0.1
    min_value_pool=3
    max_value_pool=9
    step_pool=1
    min_value_extra=16
    max_value_extra=128
    step_extra=8
    values_learning_rate = np.logspace(np.log10(0.00001), np.log10(0.1), num=20) #logarithm
    max_trials=100 # Is it like the sample size sampled from the hyperparameter space?
    executions_per_trial=2 # Is it like K-folds?
    epochs=50
    patience=16


# initialize hyperparameters
hp =  kt.HyperParameters()
    

# function filled with dynamic hyperparameters (it tries different 
# number of layers)
def build_model(hp):
    
    """Function used by Keras-tuner for doing hyperparameter tuning.
      Args:
        hp: set of hyperparameters' values to search for.   
        
      Returns:
        model: model to be fed in Keras-tuner. 
    """
    
    # initialize model
    model = tf.keras.models.Sequential()
    
    # generate layer 01
    model.add(tf.keras.layers.Conv1D(filters=hp.Int('filters_for_layer_gr_01',
                                                    min_value=min_value_filters_gr_01,
                                                    max_value=max_value_filters_gr_01,
                                                    step=step_filters_gr_01),
                                kernel_size=hp.Int('kernel_gr_01',
                                                    min_value=min_value_kernel_gr_01,
                                                    max_value=max_value_kernel_gr_01,
                                                    step=step_kernel_gr_01), 
                                padding='same', 
                                strides=1,
                                # strides=hp.Int('strides_gr_01',
                                #                                             min_value=min_value_strides,
                                #                                             max_value=max_value_strides,
                                #                                             step=step_strides),
                                input_shape=(X_train.columns.shape[0],1),
                                name='layer_gr_01'))

    model.add(tf.keras.layers.BatchNormalization())  
    # add activation layer        
    model.add(tf.keras.layers.Activation('relu'))
    
    #add dropout layer    
    model.add(tf.keras.layers.Dropout(rate=hp.Float(name='dropout_gr_01',
                                            min_value=min_value_dropout_gr_01,
                                            max_value=max_value_dropout_gr_01,
                                            step=step_dropout_gr_01), 
            name='dropout_gr_01'))
    
    # generate layer 02
    model.add(tf.keras.layers.Conv1D(filters=128,
                                kernel_size=hp.Int('kernel_gr_02',
                                                    min_value=min_value_kernel_gr_02,
                                                    max_value=max_value_kernel_gr_02,
                                                    step=step_kernel_gr_02), 
                                padding='same', 
                                strides=1,
                                name='layer_gr_02'))
    model.add(tf.keras.layers.BatchNormalization())  
    # add activation layer        
    model.add(tf.keras.layers.Activation('relu'))  
    
    #add dropout layer    
    model.add(tf.keras.layers.Dropout(rate=hp.Float(name='dropout_gr_02',
                                            min_value=min_value_dropout_gr_02,
                                            max_value=max_value_dropout_gr_02,
                                            step=step_dropout_gr_02), 
            name='dropout_gr_02'))
    
    # add Max Pooling layer
    model.add(tf.keras.layers.MaxPooling1D(pool_size=hp.Int('pool_gr_01',
                        min_value=min_value_pool,
                        max_value=max_value_pool,
                        step=step_pool),
            name='pool_gr_01'))
        
    # generate layer 03
    model.add(tf.keras.layers.Conv1D(filters=128,
                                kernel_size=hp.Int('kernel_gr_03',
                                                    min_value=min_value_kernel_gr_03,
                                                    max_value=max_value_kernel_gr_03,
                                                    step=step_kernel_gr_03), 
                                padding='same', 
                                strides=1,
                                name='layer_gr_03'))
    model.add(tf.keras.layers.BatchNormalization())  
    # add activation layer        
    model.add(tf.keras.layers.Activation('relu')) 
    
    #add dropout layer    
    model.add(tf.keras.layers.Dropout(rate=hp.Float(name='dropout_gr_03',
                                            min_value=min_value_dropout_gr_03,
                                            max_value=max_value_dropout_gr_03,
                                            step=step_dropout_gr_03), 
            name='dropout_gr_03'))
    
    # generate layer 04
    model.add(tf.keras.layers.Conv1D(filters=128,
                                kernel_size=hp.Int('kernel_gr_04',
                                                    min_value=min_value_kernel_gr_04,
                                                    max_value=max_value_kernel_gr_04,
                                                    step=step_kernel_gr_02), 
                                padding='same', 
                                strides=1,
                                name='layer_gr_04'))
        
    model.add(tf.keras.layers.BatchNormalization())  
    # add activation layer        
    model.add(tf.keras.layers.Activation('relu'))   
    
    #add dropout layer    
    model.add(tf.keras.layers.Dropout(rate=hp.Float(name='dropout_gr_04',
                                            min_value=min_value_dropout_gr_04,
                                            max_value=max_value_dropout_gr_04,
                                            step=step_dropout_gr_04), 
            name='dropout_gr_04'))
           
    # add flattening layer   
    model.add(tf.keras.layers.Flatten())
    
    # add extra  Dense layer
    model.add(tf.keras.layers.Dense(units=hp.Int('extra_dense_layer',
                                            min_value=min_value_extra,
                                            max_value=max_value_extra,
                                            step=step_extra), 
                                    name='extra_dense_layer'))
    
    #add dropout layer    
    model.add(tf.keras.layers.Dropout(rate=hp.Float(name='dropout_gr_05',
                                            min_value=min_value_dropout_gr_05,
                                            max_value=max_value_dropout_gr_05,
                                            step=step_dropout_gr_05), 
            name='dropout_gr_05'))

    # add softmax layer
    model.add(tf.keras.layers.Dense(units=len(pd.unique(y_train)), 

                                    activation='softmax', 
                                    name='output'))
       
    # compile model  
    model.compile(optimizer=tf.keras.optimizers.AdamW(
        learning_rate=hp.Choice(name='learning_rate', values=list(values_learning_rate)),
        weight_decay=hp.Float('weight_decay', min_value=1e-6, max_value=1e-2, sampling='LOG')
    ), 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

    return model

                
# Initialize Hyperband
tuner = kt.Hyperband(
    build_model,  # Function that builds the CNN
    objective='val_accuracy',
    max_epochs=epochs,  # Maximum number of epochs per trial (adjustable)
    factor=3,  # Each cycle eliminates 2/3 of the worst-performing models
    executions_per_trial=executions_per_trial,  
    directory=os.path.sep.join([BASE_DIR_OUTPUT, '{}hyperparameters_tmp'.format(output_file_name_28)]),
    project_name='multi_layer_perceptron',
    overwrite=False,
    seed=random_state
)

# Start the search with Hyperband
tuner.search(ds_train,
              epochs=epochs,  # Should be at least equal to max_epochs for proper tuning
              batch_size=batch_size,
              validation_data=ds_validation,
              callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy', 
                                                          patience=patience)])


###############################################################################
## 7. SAVE BEST SCORES IN A PANDAS DATAFRAME 

log("=== BEST HYPERPARAMETERS FOUND ===", False)
# Cattura output di search_space_summary
buffer = io.StringIO()
sys.stdout = buffer
tuner.search_space_summary()
sys.stdout = sys.__stdout__
search_summary = buffer.getvalue()
log(search_summary, False)

# Cattura output di results_summary
buffer = io.StringIO()
sys.stdout = buffer
tuner.results_summary()
sys.stdout = sys.__stdout__
results_summary = buffer.getvalue()
log(results_summary, False)
log("=================================\n", False)

# get best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

# convert Dictionary to Pandas DataFrame
best_score_tmp = pd.DataFrame.from_dict(best_hyperparameters.values, 
                                    orient = 'index')

# transpose Pandas DataFrame
summary_table = best_score_tmp.transpose()

# save cleaned best hyperparameters DataFrame as .csv file
summary_table.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                        ('{}analysis_best_scores.csv'.format(
                                        output_file_name_28))]), 
                     index = False)

# remove directory with temporary hyperparameter search trials
shutil.rmtree(os.path.sep.join([BASE_DIR_OUTPUT, 
                                           ('{}hyperparameters_tmp'.format(
                                           output_file_name_28))]))


# end time according to computer clock
end_time = time.time()

# calculate total execution time
total_execution_time = pd.Series(np.round((end_time - start_time), 2)).rename('total_hp_tuning_runtime_seconds')

# save folder's name as .csv file
total_execution_time.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                            ('{}total_hp_tuning_time.csv'.format(
                                            output_file_name_28))]), 
                            index = False)

log("=== HYPERPARAMETERS TUNING FINISHED ===\n")
# shows run-time's timestamps + total execution time
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