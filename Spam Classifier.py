import pandas as pd
import email
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

true_label = [0 for i in range(2501)]

def parse_label():
    with open('./spam-mail.tr.label', 'r') as f:
        next(f)
        i = 0
        for line in f:
             type = line.split(',')[1]
             true_label[i] = int(type)
             i += 1
    
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

def write_to_csv(sub, payload, label):
    with open("emails.csv", "a", encoding="utf-8") as f:
        payload = payload.replace('\n', "")
        writer = csv.writer(f)
        writer.writerow([sub, payload, label])

def write_header():
    header = ["subject", "email", "class"]
    with open("emails.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def parse_emails():
    parse_label()
    write_header()

    for i in range(1, 2501):
        path = ".\TR\TRAIN_%d.eml" % i
        read_file(path, i - 1)


if __name__ == "__main__":
    parse_emails()
    
    data = pd.read_csv("./emails.csv")

    # Create a vocabulary
    vectorizer = CountVectorizer(stop_words='english')
    all_features = vectorizer.fit_transform(data['email'])

    # Split the data
    email_train, email_test, class_train, class_test = train_test_split(all_features, data['class'], test_size=0.3)

    # Train and predict
    clf = MultinomialNB()
    clf.fit(email_train, class_train)
    predictions = clf.predict(email_test)
    print(classification_report(class_test, predictions))
    print(confusion_matrix(class_test, predictions))