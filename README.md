# Email-Spam-Classifier

# Description
- Take in a bunch of emails where some are marked as spam and others are not. Then train the classifier to predict which one is spam or not using Naive Bayes. It is currently identifying the correct type of email (Spam / Not spam) with an accuracy of roughly 95%.

(1 = spam, 0 = not spam)
![image](https://user-images.githubusercontent.com/60799172/134743193-cf3fd512-9f60-4e15-9e80-540f24fe40c6.png)

# How Naive Bayes Works Here
- I calculated the probabilites for the words in spam and not spam into a dictionary. I did this because I wanted to find P(word | spam) and P(word | not spam). 
- Next, I calculated a prediction (1 or 0) by using the probabilities I created earlier. I did this by calculating P(spam | words) and P(not spam | words). Whichever one gave me the higher probability is the one I went with. 
- P(spam | words) = (ln(P(word1 | spam)) + ln(P(word2 | spam)) + ... + ln(P(wordn | spam))) + ln(P(spam))
- P(not spam | words) = (ln(P(word1 | not spam)) + ln(P(word2 | not spam)) + ... + ln(P(wordn | not spam))) + ln(P(not spam))
