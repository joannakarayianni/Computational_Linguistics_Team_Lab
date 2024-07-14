import pandas as pd
from scripts.evaluation.vaulting import Vaulting


# Loading prediction files for all the models
sequential_nn_glove = pd.read_csv('scripts/advanced_classifier/sequential_nn/predictions/predictions_seq_nn_glove.csv')

# Golden standard (test data)
ground_truth_test = pd.read_csv('scripts/evaluation/ground_truths/ground_truth_test.csv')

# Models list
models = {
    'Sequential NN with Glove': sequential_nn_glove,
}


vaulting_instance = Vaulting(ground_truth_test, models)
vaulting_instance.findResults()