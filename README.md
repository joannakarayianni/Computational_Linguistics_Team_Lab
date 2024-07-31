# Emotion Text Detection Project

This project is of the course Computational Linguistics Team Laboratory for the MSc in Computational Linguistics at the University of Stuttgart.

# Description

Emotion text detection involves classifying text based on emotional content, a critical and challenging aspect of Natural Language Processing (NLP) due to the nuanced nature of human emotions. In this project, we aimed to perform a multilabel classification on the ISEAR dataset (CISA, University of Geneva, 2024-07-16), which includes 7 classes - Joy, Sadness, Disgust, Guilt, Shame, Fear, and Anger. It consists of textual descriptions annotated with these emotion labels. The primary research question we aimed to address is: "How can different machine learning models and embedding techniques, which capture contextual information, be utilized to effectively classify fine-grained emotional content in text?"

## Results from experiments

![Table Image](https://github.com/joannakarayianni/Computational_Linguistics_Team_Lab/blob/main/images/performance_comparison.png)

# Setup

## Create the virtual environment:-

python -m venv venv       
source venv/bin/activate     


## Install the project and the dependencies using:- 

pip install .


## Make sure that the datasets are present in a folder named datasets at the root level.

|-- config  
|-- scripts    
|-- dataanalysis   
|-- datasets   

## Ensure that you have word embeddings downloaded to run the GloVe based Seq. Neural Networks & LSTM GloVe in the below locations.
Embeddings were downloaded from this website: https://nlp.stanford.edu/projects/glove/ , 'glove.6B.zip'

Locations:

Seq. NNs: scripts/advanced_classifier/word_embeddings/glove.6B/glove.6B.300d.txt

LSTM GloVe: scripts/advanced_classifier/lstm/glove.6B.100d.txt

## To visualize the performance of various models for each metric, execute the following command:
#### Note: A similar visualization process has been applied for the sequence neural network models as well.

python3 -m scripts.advanced_classifier.lstm.lstm_model_comparison_visualizer


## Run

python ./scripts/main.py
