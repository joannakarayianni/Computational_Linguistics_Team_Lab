import math
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

# get the list of English stop words
stop_words = set(stopwords.words('english'))
class NaiveBayes:

    def __init__(self):
        # Datastructures for Naive Bayes Classifier. 
        self.class_counts = defaultdict(int) # dictionary of class vs count of each class in the training set.
        self.term_counts_dictionary = defaultdict(lambda: defaultdict(int)) # dictionary that maps an emotion class to a set of terms and their respective frequencies.
        self.N = 0 # Total number of records in the dataset.
        self.vocab = set() # set of unique words in the training dataset.
        self.prior_probs = defaultdict(int)
        self.likelihood_probs = defaultdict(lambda: defaultdict(float))
        

    def construct_dictionary_and_vocab(self, df_train):
        for index, row in df_train.iterrows():

            # unpacking the row fields.
            emotion_class, text = row

            # tokenize the text.
            tokens = word_tokenize(text)

            # cleanse the tokens
            filtered_tokens = self.preprocess_tokens(tokens, stop_words)
                        
            self.class_counts[emotion_class] += 1

            for token in filtered_tokens:
                term = self.normalizeTerm(token)
                # Skip if the term is None (punctuation or empty)
                if term is None:
                    continue
                #construct the vocabulary of unique terms
                self.vocab.add(term)
                # update the term count for the given class
                self.term_counts_dictionary[emotion_class][term] += 1

        # we sum up all the review classes seen in the records to arrive at N. 
        self.N = sum(self.class_counts.values())

    def preprocess_tokens(self, tokens, stop_words):
        # remove stop words and duplicates from the tokenized list
        filtered_tokens = set(tokens) - stop_words
        
        # convert the set back to a list
        return list(filtered_tokens)

    
    def normalizeTerm(self, term):
        # convert the term to lower case
        term = term.lower()

        # Check if the term is a punctuation or empty
        if term == '' or term in string.punctuation:
            return None
        
        return term
    
    def train_the_classifer(self):
        # the below probabilities are estimated probabilities since they are computed using the training set.
        prior_probs = defaultdict(int)

        # compute the prior probability for each class.
        for review_class, count in self.class_counts.items():
            prior_prob_for_given_class = count / self.N
            prior_probs[review_class] = prior_prob_for_given_class
        
        likelihood_probs = defaultdict(lambda: defaultdict(float))

        # for each class, compute the conditional probability for each term in the vocabulary.
        for cls, term_counts in self.term_counts_dictionary.items():
            tct_dashed = sum(term_counts.values())
            for term in self.vocab:
                tct = term_counts[term]
                likelihood_probs[cls][term] = (tct + 1) / (tct_dashed + len(self.vocab))
            # introducing a default likelihood probability for each class that would be retreived when the term is not present in the map.
            likelihood_probs[cls]['default'] = (1) / (tct_dashed + len(self.vocab))
        
        self.prior_probs = prior_probs
        self.likelihood_probs = likelihood_probs
        print(self.likelihood_probs['joy'])
    
    def get_the_best_class(self, text):
        tokens = word_tokenize(text)
        # cleanse the tokens
        filtered_tokens = self.preprocess_tokens(tokens, stop_words)
        class_scores = defaultdict(float)
        
        for cls in self.prior_probs.keys():
            class_scores[cls] = math.log(self.prior_probs[cls])  # Initialize with prior probability
            for term in filtered_tokens:
                term = self.normalizeTerm(term)
                # Skip if the term is None (punctuation or empty)
                if term is None:
                    continue
                if term not in self.likelihood_probs[cls]:
                    term = 'default'
                class_scores[cls] += math.log(self.likelihood_probs[cls][term])

        # use maximum a posteriori to get the best class.
        return max(class_scores, key=class_scores.get)