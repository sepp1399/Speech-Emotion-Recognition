"""
TITLE: "Analysis Script for Model Results: Comparing 'Talking About' with Dataset Ground Truth"
AUTHORS: Giuseppe Lentini
LAST UPDATE: 20250219
PYTHON VERSION: 3.10.6

DESCRIPTION: 
Further, please change the following sections according to your individidual input preferences:
    - 'SETTING PATHS AND KEYWORDS'; 
    - 'PARAMETERS TO BE SET!!!'
    - TRICK: at each run of the script below re-start the Python kernel (e.g. re-start Spyder or PyCharm)

"""

###############################################################################
## 1. IMPORTING LIBRARIES
# import required Python libraries
import platform
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, recall_score
from datetime import datetime

###############################################################################
## 2. PARAMETERS TO BE SET!!!

# check the machine you are using 
RELEASE = platform.release()

# set the correct pathways/folders
BASE_DIR_INPUT = (".")
BASE_DIR_OUTPUT = BASE_DIR_INPUT
INPUT_FOLDER = ("./..")
BASE_DATASET = 'Emozionalmente_augmented_core'

# set input/output file names set input/output file names
model_number = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         'config/model_number.csv']), header = None, 
                                     dtype = 'str').iloc[0].values[0]
input_file_name_05 = ("config/run_modality.csv")
output_file_name_28 = (f'output/{model_number}/')
output_file_name_29 = ("input/")

model_number = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         'config/model_number.csv']), header = None, 
                                     dtype = 'str').iloc[0].values[0]

###############################################################################
## 3. GENERAL SETUP: FILES PREPARATION

print("#"*70) # Section: General Setup

# Load the evaluations file
evaluations_file_path = "{}datasets/{}/metadata/evaluations.csv".format(output_file_name_29, BASE_DATASET)

evaluations_df = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, evaluations_file_path]), header=0, dtype='str')

# Construct the results file name with the current date
results_file_name = "{}results/Talking_About_{}_results.csv".format(
    output_file_name_28, model_number)
results_file_path = os.path.join(BASE_DIR_OUTPUT, results_file_name)

# Load the results file
try:
    results_df = pd.read_csv(results_file_path, header=0, dtype='str')
    print(f"Loaded file: {results_file_name}")
except FileNotFoundError:
    print(f"Error: File '{results_file_name}' not found. Specify a correct date.")

###############################################################################
## 4. DATA CLEANING

# these are the relevant columns in the evaluations file for the analysis
evaluations_df = evaluations_df.filter(items=['emotion_expressed', 'file_name'])

# Define the emotions to keep for 'Talking About'
emotions = ["angry", "disgust", "happy", "sad", "fear", 'surprise', 'calm']

# Calculate the mode for each file in the evaluations_df
evaluations_df = (
    evaluations_df.groupby("file_name")["emotion_expressed"]
    .agg(lambda x: x.value_counts().idxmax())  #this way to prevent multiple mode (we get the 1°)
    .reset_index()
)
# Filter the DataFrame
evaluations_df_filtered = evaluations_df[evaluations_df["emotion_expressed"].isin(emotions)]
results_df = results_df[results_df["emotion_prediction"].isin(emotions)]

# merge results & evaluations datasets
merged_df = pd.merge(evaluations_df_filtered, results_df, left_on="file_name", right_on="file_name", how="inner")

# Remove values with probability >= 0.66
merged_df['probability'] = pd.to_numeric(merged_df['probability'], errors='coerce')
merged_df = merged_df[merged_df['probability'] >= 0.66]

# Remove potential missing values and duplicated rows
merged_df.drop_duplicates(inplace=True)
merged_df.dropna(inplace=True)
###############################################################################
## 5. DATA EXPLORATION AND ANALYSIS

print("_"*70+"\n") # Section: Output info to terminal

# Create output directory if it doesn't exist
output_dir = "output/{}/analysis".format(model_number)
os.makedirs(output_dir, exist_ok=True)

# Save the input data of the analysis
merged_df_csv_file = os.path.join(output_dir, "Talking_about_analysis.csv")

# define columns order
merged_df = merged_df[['file_name', 'emotion_expressed', 'emotion_prediction', 'probability']]

# Save the results base info to the CSV file
merged_df.to_csv(merged_df_csv_file, index=False)

# Specify the output file paths
results_file = os.path.join(output_dir, "{}_analysis.txt".format(model_number))
conf_matrix_file = os.path.join(output_dir, "confusion_matrix.png")
probabilities_file = os.path.join(output_dir, "prediction_probabilities.png")

# Open a file to write the analysis results
with open(results_file, "w") as f:
    # Write basic summary statistics of the dataset
    f.write("Basic summary statistics:\n")
    f.write(str(merged_df.describe()) + "\n\n")
    
    # Write the distribution of emotions in the ground truth
    f.write("Emotion distribution in 'emotion_expressed' (Ground Truth):\n")
    f.write(str(merged_df['emotion_expressed'].value_counts()) + "\n\n")
    
    # Write the distribution of emotions in the model predictions
    f.write("Emotion distribution in 'emotion_prediction' (Model Predictions):\n")
    f.write(str(merged_df['emotion_prediction'].value_counts()) + "\n\n")
    
    # Compute and write the confusion matrix
    conf_matrix = confusion_matrix(merged_df['emotion_expressed'], merged_df['emotion_prediction'])
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n\n")
    
    # Calculate and write the model accuracy
    accuracy = accuracy_score(merged_df['emotion_expressed'], merged_df['emotion_prediction'])
    f.write(f"Accuracy of the model: {accuracy * 100:.2f}%\n\n")

    uar = recall_score(merged_df['emotion_expressed'], merged_df['emotion_prediction'], average="macro")
    f.write(f"Unweighted Average Recall (UA): {uar:.4f}\n\n")

    # Calculate and write WA (Weighted Average Recall)
    war = recall_score(merged_df['emotion_expressed'], merged_df['emotion_prediction'], average="weighted")
    f.write(f"Weighted Average Recall (WA): {war:.4f}\n\n")

    # Calculate and write MCC score
    mcc_score = matthews_corrcoef(
        merged_df['emotion_expressed'], 
        merged_df['emotion_prediction']
    )
    f.write(f"Matthews Correlation Coefficient (MCC): {mcc_score:.4f}\n\n")

    # Generate and write the classification report
    class_report = classification_report(
        merged_df['emotion_expressed'],
        merged_df['emotion_prediction'],
        target_names = emotions
    )
    f.write("Classification Report:\n")
    f.write(class_report + "\n\n")
    
    # Calculate and write the correlation between ground truth and predictions
    correlation = merged_df['emotion_expressed'].astype('category').cat.codes.corr(
        merged_df['emotion_prediction'].astype('category').cat.codes
    )
    f.write(f"Correlation between Ground Truth and Predictions: {correlation}\n\n")
    
    # Write the performance for each emotion
    for emotion in emotions:
        emotion_data = merged_df[merged_df['emotion_expressed'] == emotion]
        emotion_accuracy = accuracy_score(emotion_data['emotion_expressed'], emotion_data['emotion_prediction'])
        f.write(f"Performance for '{emotion}' emotion:\n")
        f.write(f"Accuracy for {emotion}: {emotion_accuracy * 100:.2f}%\n\n")
     
# Save the confusion matrix plot as an image
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", 
            xticklabels=emotions, 
            yticklabels=emotions)
plt.xlabel('Predicted Emotion')
plt.ylabel('Actual Emotion')
plt.title('Confusion Matrix - Model vs Ground Truth')
plt.savefig(conf_matrix_file)
plt.close()

# Save the prediction probabilities plot as an image
plt.figure(figsize=(8, 6))
sns.boxplot(x='emotion_prediction', y='probability', data=merged_df)
plt.title('Distribution of Prediction Probabilities for Each Emotion')
plt.xlabel('Predicted Emotion')
plt.ylabel('Prediction Probability')
plt.savefig(probabilities_file)
plt.close()

print(f"Analysis saved in {results_file}\nPlots saved in {output_dir}")
print("#"*70) # end of the script

# save model performance 
output_csv_path = os.path.join('output', "performance.csv")

# Check if the file exists
if not os.path.exists(output_csv_path):
    # Create an empty DataFrame with the desired columns
    columns = ["Model", "Accuracy", "UA", "WA", "MCC"] 
    results_df = pd.DataFrame(columns=columns)
    results_df.to_csv(output_csv_path, index=False)
else:
    results_df = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, output_csv_path]), header = 0)

# Create a DataFrame with model_number and accuracy
new_row = pd.DataFrame({"Model": [model_number], "Accuracy": [f'{accuracy * 100:.2f}%'], "UA": [round(uar, 3)], "WA": [round(war, 3)], "MCC": [round(mcc_score, 3)]})
results_df = pd.concat([results_df, new_row], ignore_index=True)

# Save the DataFrame to CSV
results_df.to_csv(output_csv_path, index=False)