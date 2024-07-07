""" Run this file seperately """
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
import random

# Setting the environment variable to use legacy Keras
os.environ['TF_USE_LEGACY_KERAS'] = '1'

""" BERT model trained for multilabel emotion classification """
class BertModel:
    def __init__(self, train_path, val_path, test_path, emotions, max_length=100, seed=42):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.emotions = emotions
        self.max_length = max_length
        self.seed = seed

        # Setting seeds for securing same results in every run
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        random.seed(self.seed)

# Bert Model used 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(emotions))

        # Fit the MultiLabelBinarizer with the emotions list
        self.mlb = MultiLabelBinarizer(classes=self.emotions)
        self.mlb.fit([[emotion] for emotion in self.emotions])

        self.train_df, self.val_df, self.test_df = self.load_data()
        self.X_train, self.attention_masks_train, self.y_train = self.encode_data(self.train_df)
        self.X_val, self.attention_masks_val, self.y_val = self.encode_data(self.val_df, training=False)
        self.X_test, self.attention_masks_test, self.y_test = self.encode_data(self.test_df, training=False)

        self.model = self.build_model()

    def load_data(self):
        # Load training data, making sure text is string & emotion labels from the list are used
        train_df = pd.read_csv(self.train_path, header=None, names=['emotion', 'text'], on_bad_lines='skip')
        train_df['text'] = train_df['text'].astype(str)
        train_df = train_df[train_df['emotion'].isin(self.emotions)]

        # Load validation data, making sure text is string & emotion labels from the list are used
        val_df = pd.read_csv(self.val_path, header=None, names=['emotion', 'text'], on_bad_lines='skip')
        val_df['text'] = val_df['text'].astype(str)
        val_df = val_df[val_df['emotion'].isin(self.emotions)]

        # Load test data, making sure text is string & emotion labels from the list are used
        test_df = pd.read_csv(self.test_path, header=None, names=['emotion', 'text'], on_bad_lines='skip')
        test_df['text'] = test_df['text'].astype(str)
        test_df = test_df[test_df['emotion'].isin(self.emotions)]

        return train_df, val_df, test_df
    
# tokenizing text data into BERT-compatible input ids & attention masks
# converting emotion labels into one-hot encoded vectors
   
    def encode_data(self, df, training=True):
        encodings = self.tokenizer(df['text'].tolist(), truncation=True, padding=True, max_length=self.max_length, return_tensors='tf')
        input_ids = encodings['input_ids']
        attention_masks = encodings['attention_mask']

        labels = self.mlb.transform(df['emotion'].apply(lambda x: [x]))

        return input_ids, attention_masks, labels

# Building the model
    def build_model(self):
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')

        bert_output = self.bert_model(input_ids, attention_mask=attention_mask).logits
        output = tf.keras.layers.Activation('sigmoid')(bert_output)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

        # Using the legacy Adam optimizer
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=3e-5)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model
    
# Training the model
    def train(self, epochs=10, batch_size=32):
        # use early stopping to prevent overfittng
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(
            [self.X_train, self.attention_masks_train], self.y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=([self.X_val, self.attention_masks_val], self.y_val),
            callbacks=[early_stopping]
        )
        return history
    
# evaluation on validation set
    def evaluate_validation(self):
        loss, accuracy = self.model.evaluate([self.X_val, self.attention_masks_val], self.y_val, verbose=0)
        print(f'Validation Loss: {loss}')
        print(f'Validation Accuracy: {accuracy}')
        return loss, accuracy
# evaluation on test set
    def evaluate_test(self):
        loss, accuracy = self.model.evaluate([self.X_test, self.attention_masks_test], self.y_test, verbose=0)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')
        return loss, accuracy
    
    # predictions on validation set
    def predict_validation(self):
        y_pred = self.model.predict([self.X_val, self.attention_masks_val])
        y_pred = (y_pred > 0.5).astype(int)
        print("Results on validation set:")
        print(classification_report(self.y_val, y_pred, target_names=self.emotions))
        
    # predictions on test set
    def predict_test(self):
        y_pred = self.model.predict([self.X_test, self.attention_masks_test])
        y_pred = (y_pred > 0.5).astype(int)
        predictions_df = pd.DataFrame(y_pred, columns=self.emotions)
        predictions_df.to_csv('/Users/ioannakaragianni/Documents/GitHub/Computational_Linguistics_Team_Lab/scripts/advanced_classifier/lstm/predictionsbert.csv', index=False)
        # print report
        print("Results on the test set:")
        print(classification_report(self.y_test, y_pred, target_names=self.emotions))

# Run 
train_path = 'datasets/isear-train.csv'
val_path = 'datasets/isear-val.csv'
test_path = 'datasets/isear-test.csv'
emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']

classifier = BertModel(train_path, val_path, test_path, emotions)
classifier.train(epochs=10, batch_size=32)
classifier.evaluate_validation()
classifier.evaluate_test()
classifier.predict_validation()
classifier.predict_test()