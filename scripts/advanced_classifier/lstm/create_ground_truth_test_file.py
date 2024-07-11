""" In this script I aimed to transform the isear test dataset (which will be my golden standard) in the same format as the predictions
csv's for each one of my 5 models so I can afterwards make comparisons according to them"""
import pandas as pd

# List of emotions
emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']

# ground test truth data to transform to 0-1 format like the predictions
ground_truth = pd.read_csv('datasets/isear-test.csv', names=['Emotion', 'Text'])

# empty DataFrame with the required structure
ground_truth_binary = pd.DataFrame(0, index=ground_truth.index, columns=emotions)

# Populating the DataFrame
for idx, row in ground_truth.iterrows():
    emotion = row['Emotion'].strip()
    if emotion in emotions:
        ground_truth_binary.at[idx, emotion] = 1

# Save to a new CSV file
ground_truth_binary.to_csv('ground_truth.csv', index=False)

print(ground_truth_binary.head())



