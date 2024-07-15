import os
import pandas as pd
import numpy as np
from transformers import DebertaTokenizer, TFAutoModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
import random

# Setting the environment variable to use legacy Keras
os.environ['TF_USE_LEGACY_KERAS'] = '1'

class DebertaLSTMModel:
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

        self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        self.deberta_model = TFAutoModel.from_pretrained('microsoft/deberta-base')

        self.mlb = MultiLabelBinarizer(classes=self.emotions)
        self.mlb.fit([[emotion] for emotion in self.emotions])

        self.train_df, self.val_df, self.test_df = self.load_data()
        self.X_train, self.attention_masks_train, self.y_train = self.encode_data(self.train_df)
        self.X_val, self.attention_masks_val, self.y_val = self.encode_data(self.val_df, training=False)
        self.X_test, self.attention_masks_test, self.y_test = self.encode_data(self.test_df, training=False)

        self.model = self.build_model()

    def load_data(self):
        train_df = pd.read_csv(self.train_path, header=None, names=['emotion', 'text'], on_bad_lines='skip')
        train_df['text'] = train_df['text'].astype(str)
        train_df = train_df[train_df['emotion'].isin(self.emotions)]

        val_df = pd.read_csv(self.val_path, header=None, names=['emotion', 'text'], on_bad_lines='skip')
        val_df['text'] = val_df['text'].astype(str)
        val_df = val_df[val_df['emotion'].isin(self.emotions)]

        test_df = pd.read_csv(self.test_path, header=None, names=['emotion', 'text'], on_bad_lines='skip')
        test_df['text'] = test_df['text'].astype(str)
        test_df = test_df[test_df['emotion'].isin(self.emotions)]

        return train_df, val_df, test_df

    def encode_data(self, df, training=True):
        encodings = self.tokenizer(df['text'].tolist(), truncation=True, padding=True, max_length=self.max_length, return_tensors='tf')
        input_ids = encodings['input_ids']
        attention_masks = encodings['attention_mask']

        labels = self.mlb.transform(df['emotion'].apply(lambda x: [x]))

        return input_ids, attention_masks, labels

    def build_model(self):
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')

        # Extract embeddings from DeBERTa
        embeddings = self.deberta_model(input_ids, attention_mask=attention_mask).last_hidden_state

        # Add LSTM layers and dropout
        lstm_output1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(embeddings)
        dropout1 = tf.keras.layers.Dropout(0.2)(lstm_output1)
        lstm_output2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(dropout1)

        output = tf.keras.layers.Dense(len(emotions), activation='sigmoid')(lstm_output2)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def train(self, epochs=10, batch_size=32):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(
            [self.X_train, self.attention_masks_train], self.y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=([self.X_val, self.attention_masks_val], self.y_val),
            callbacks=[early_stopping]
        )
        return history

    def evaluate_validation(self):
        loss, accuracy = self.model.evaluate([self.X_val, self.attention_masks_val], self.y_val, verbose=0)
        print(f'Validation Loss: {loss}')
        print(f'Validation Accuracy: {accuracy}')
        return loss, accuracy

    def evaluate_test(self):
        loss, accuracy = self.model.evaluate([self.X_test, self.attention_masks_test], self.y_test, verbose=0)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')
        return loss, accuracy

    def predict_validation(self):
        y_pred = self.model.predict([self.X_val, self.attention_masks_val])
        y_pred = (y_pred > 0.5).astype(int)
        print("Results on validation set:")
        print(classification_report(self.y_val, y_pred, target_names=self.emotions))

    def predict_test(self):
        y_pred = self.model.predict([self.X_test, self.attention_masks_test])
        y_pred = (y_pred > 0.5).astype(int)
        predictions_df = pd.DataFrame(y_pred, columns=self.emotions)
        predictions_path = 'scripts/advanced_classifier/lstm/predictions/predictions_lstm_deberta.csv'
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        predictions_df.to_csv(predictions_path, index=False)
        print("Results on the test set:")
        print(classification_report(self.y_test, y_pred, target_names=self.emotions))

# Run 
train_path = 'datasets/isear-train.csv'
val_path = 'datasets/isear-val.csv'
test_path = 'datasets/isear-test.csv'
emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']

classifier = DebertaLSTMModel(train_path, val_path, test_path, emotions)
classifier.train(epochs=10, batch_size=32)
classifier.evaluate_validation()
classifier.evaluate_test()
classifier.predict_validation()
classifier.predict_test()
