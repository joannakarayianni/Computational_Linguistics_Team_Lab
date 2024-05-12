class EmotionSample:
    def __init__(self, emotions, text):
        self.emotions = emotions
        self.text = text
        self.features = self.extract_features() # Splitting text into features

    def extract_features(self):
        return self.text.split()