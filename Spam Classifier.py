import pandas as pd
import email
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from NaiveBayes import NaiveBayes as NB

true_label = [0 for i in range(2501)]

def parse_label():
    with open('./spam-mail.tr.label', 'r') as f:
        next(f)
        i = 0
        for line in f:
             type = line.split(',')[1]
             true_label[i] = int(type)
             i += 1
    
def read_file(filename, id, t):
    with open(filename, encoding='latin-1') as fp:
        msg = email.message_from_file(fp)
        payload = msg.get_payload()
        if type(payload) == type(list()):
            payload = payload[0]
        if type(payload) != type(''):
            payload = str(payload)

        sub = msg.get('subject')
        sub = str(sub)
        
        write_to_csv(sub,payload,true_label[id], t)

def write_to_csv(sub, payload, label, t):
    if t == "TT":
        name = "test.csv"
    else:
        name = "emails.csv"

    with open(name, "a", encoding="utf-8") as f:
        payload = payload.replace('\n', "")
        writer = csv.writer(f)
        writer.writerow([sub, payload, label])

def write_header(t):
    if t == "TT":
        name = "test.csv"
    else:
        name = "emails.csv"

    header = ["subject", "email", "class"]
    with open(name, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def parse_emails():
    parse_label()
    write_header("TR")
    write_header("TT")

    for i in range(1, 2501):
        path = ".\TR\TRAIN_%d.eml" % i
        read_file(path, i - 1, "TR")
        
    for i in range(1, 1828):
        path = ".\TT\TEST_%d.eml" % i
        read_file(path, i - 1, "TT")


if __name__ == "__main__":
    parse_emails()
    
    data = pd.read_csv("./emails.csv")
    #test = pd.read_csv("./test.csv")

    X_train, X_test, y_train, y_test = train_test_split(data['email'], data['class'], test_size=0.3)
    clf = NB()
    clf.fit(X_train, np.array(y_train))
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    '''
    with open("testLabels.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prediction"])
        for i in range(1, len(y_pred) + 1):
            writer.writerow([i, y_pred[i-1]])
   '''