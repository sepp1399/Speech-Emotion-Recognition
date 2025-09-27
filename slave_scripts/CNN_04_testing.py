"""
TITLE: ""
AUTHOR: Giuseppe Lentini 
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
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
import time
import onnxruntime as rt

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

################################################################################
## 3. PREDICTION AND RESULTS SAVING

# start clocking time
start_time = time.time()

# load test set
test_set = pd.read_csv(os.path.join("output", model_number, "data", "test_set.csv"))

# Construct the results file name with the current date
results_file_name = "{}results/Talking_About_{}_results.csv".format(
    output_file_name_28, model_number)


# Construct the directory path
results_dir = os.path.sep.join([BASE_DIR_OUTPUT, f"output/{model_number}/results/"])

# Ensure the directory exists
os.makedirs(results_dir, exist_ok=True)

# Define the CSV columns
columns = ["file_name", "emotion_prediction", "probability"]

# Create an empty DataFrame and save it
results_df = pd.DataFrame(columns=columns)
results_df.to_csv(results_file_name, index=False)

replace_emotions = {0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sad', 6: 'surprise'}
# replace_emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise'}

# initialize session
sess_options = rt.SessionOptions()

sess = rt.InferenceSession(os.path.sep.join([BASE_DIR_OUTPUT, 
                                                     ('output/{}/saved_model/best_hyper_model.onnx'.format(model_number))]), sess_options)

# generate predictions
def make_predictions(test_set):
          
    # re-shape data-set
    x_test = np.expand_dims(test_set, axis=2)

    # set inputs/outputs
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    # generate predictions on the new data
    onnx_predictions = sess.run(
        [label_name], {input_name: x_test.astype(np.float32)})[0]

    # build recognitions from test-set by using trained model
    recognition_tmp = pd.DataFrame(data = np.around(a=onnx_predictions, decimals=3), 
                                    columns = list(replace_emotions.values()))
    
    return recognition_tmp

for audio in test_set.itertuples(index=False):

    audio_features = np.array(audio[:-3]).reshape(1, -1)
    recognition = pd.DataFrame(make_predictions(audio_features))

    # Extract the prediction and the probability
    prediction = recognition.iloc[0, :].idxmax()  # Find the class with the maximum value
    probability = recognition.iloc[0, :].max()  # Get the maximum value (probability)

    # Create a new record to be added
    new_data = {"file_name": audio.file_name, "emotion_prediction": prediction, "probability": probability}

    # Create a DataFrame for the new row and concatenate it with the existing DataFrame
    new_row = pd.DataFrame([new_data])
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Save the updated file
    results_df.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, results_file_name]), index=False)


# end time according to computer clock
end_time = time.time()

# calculate total execution time
total_execution_time = pd.Series(np.round((end_time - start_time), 2)).rename('total_training_runtime_seconds')

# shows run-time's timestamps + total execution time
print('start time (unix timestamp):{}'.format(start_time))
print('end time (unix timestamp):{}'.format(end_time))
print('total execution time (seconds):{}'.format(total_execution_time.iloc[0]))