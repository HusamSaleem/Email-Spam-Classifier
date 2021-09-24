import numpy as np
from nltk.corpus import stopwords

class NaiveBayes():
    # Takes the data and parses it into spam and not spam probability
    def fit(self, X, y):
        self.parse_emails(X, y)
        pass

    # Initial guess * words probs
    def predict(self, X):
        y_pred = [0 for i in range(len(X))]

        initial_spam_guess = (self._spam_words_count) / (self._spam_words_count + self._not_spam_words_count)
        initial_not_spam_guess = (self._not_spam_words_count) / (self._not_spam_words_count + self._spam_words_count)

        i = 0
        for x in X:
            email = self.get_as_arr(x)

            spam_score = 0
            notspam_score = 0

            for word in email:
                if not word in self._spam_words:
                    # Laplace Smoothing
                    spam_score += np.log(1/self._spam_words_count)
                else:
                    spam_score += np.log(self._spam_words[word])

                if not word in self._not_spam_words:
                    # Laplace Smoothing
                    notspam_score += np.log(1/self._not_spam_words_count)
                else:
                    notspam_score += np.log(self._not_spam_words[word])

            spam_score += np.log(initial_spam_guess)
            notspam_score += np.log(initial_not_spam_guess)

            if (spam_score > notspam_score):
                y_pred[i] = 1
            else:
                y_pred[i] = 0
            
            i += 1
        return y_pred

    def get_as_arr(self, x):
        stop_words = set(stopwords.words('english'))
        word_tokens = x.split()
        clean_sentence = [w for w in word_tokens if not w.lower() in stop_words]

        return clean_sentence

    def parse_emails(self, X, y):
        # Contain the word and their probability
        self._spam_words = {}
        self._not_spam_words = {}

        i = 0
        for email in X:
            clean_sentence = self.get_as_arr(email)

            for word in clean_sentence:
                if (y[i] == 0):
                   if not word in self._not_spam_words:
                        self._not_spam_words[word] = 1
                   else:
                        self._not_spam_words[word] += 1
                else:
                    if not word in self._spam_words:
                        self._spam_words[word] = 1
                    else:
                        self._spam_words[word] += 1
            
            i += 1
        self._spam_words_count = len(self._spam_words)
        self._not_spam_words_count = len(self._not_spam_words)

        self.calculate_word_probability()
    
    def calculate_word_probability(self):
        for word in self._not_spam_words:
            self._not_spam_words[word] = (self._not_spam_words[word] / self._not_spam_words_count)

        for word in self._spam_words:
            self._spam_words[word] = (self._spam_words[word] / self._spam_words_count)