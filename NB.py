import numpy as np
import pandas as pd
from pathlib import Path
import email
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import string
from bs4 import BeautifulSoup
import csv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix

true_label = [0 for i in range(2500)]

def parse_label():
    with open('./spam-mail.tr.label', 'r') as f:
        next(f)
        i = 0
        for line in f:
             type = line.split(',')[1]
             true_label[i] = int(type)
             i += 1
    
'''
use email to process the email content
:param filename: email path
:return: email title and content
'''
def read_file(filename, id):
    with open(filename, encoding='latin-1') as fp:
        msg = email.message_from_file(fp)
        payload = msg.get_payload()
        if type(payload) == type(list()):
            payload = payload[0]
        if type(payload) != type(''):
            payload = str(payload)

        sub = msg.get('subject')
        sub = str(sub)
        
        write_to_csv(sub,payload,true_label[id])

def write_header():
    header = ["subject", "email", "class"]
    with open("emails.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def write_to_csv(sub, payload, label):
    with open("emails.csv", "a", encoding="utf-8") as f:
        payload = payload.replace('\n', "")
        cleantext = BeautifulSoup(payload, features="html.parser").text # Clean the emails of html tags
        writer = csv.writer(f)
        writer.writerow([sub, payload, label])

def text_process(email):
    nopunc = [char for char in email if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


if __name__ == "__main__":
    training_data_directory = ".\TR"
    parse_label()

    write_header()
    id = 0
    # Read the training data emails
    for filename in os.listdir(training_data_directory):
        email_path = os.path.join(training_data_directory, filename)
        read_file(email_path, id)
        id += 1
    
    nltk.download('stopwords')
    df_train = pd.read_csv("./emails.csv")

    email_train, email_test, class_train, class_test = train_test_split(df_train['email'], df_train['class'], test_size=0.3)
    
    pipeline = Pipeline([( 'bow',CountVectorizer(analyzer=text_process)), ('tfidf',TfidfTransformer()), ('classifier',MultinomialNB()),])
    pipeline.fit(email_train, class_train)
    predictions = pipeline.predict(email_test)
    print(classification_report(predictions, class_test))
    
    '''
    # Vectorizing the data
    bow_transformer = CountVectorizer(analyzer=text_process).fit(email_train)
    messages_bow = bow_transformer.transform(email_train)

    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)

    # Training
    spam_detect_model = MultinomialNB().fit(messages_tfidf, class_train)
    all_predictions = spam_detect_model.predict(email_test)

    print(all_predictions)
    print(classification_report(class_test, all_predictions))
    print(confusion_matrix(class_test,all_predictions))
    '''