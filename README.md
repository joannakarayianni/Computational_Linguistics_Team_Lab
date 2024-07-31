# Emotion Text Detection Project

This project is of the course Computational Linguistics Team Laboratory for the MSc in Computational Linguistics at the University of Stuttgart.


## Create the virtual environment:-

python -m venv venv       
source venv/bin/activate     


## Install the project and the dependencies using:- 

pip install .


## Make sure that the datasets are present in a folder named datasets at the root level

|-- config  
|-- scripts    
|-- dataanalysis   
|-- datasets   

## Ensure that you have word embeddings downloaded to run the GloVe based Seq. Neural Networks.
 
|-- scripts    
    |-- advanced_classifier   
        |-- word_embeddings  
            |-- glove.6B
                    |--glove.6B.300d.txt

## Run

python ./scripts/main.py
