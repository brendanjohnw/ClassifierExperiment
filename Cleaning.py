import pandas as pd
import nltk
import string
import re
from nltk.corpus import stopwords

train_data = pd.read_csv("Train.csv", sep=',')
test_data = pd.read_csv("Test.csv", sep=',')
X_train_raw = [x[0] for x in train_data[['text']].values]
Y_train = [x[0] for x in train_data[['sentiment']].values]
X_test_raw = [x[0] for x in test_data[['text']].values]
X_train_clean = []
X_test_clean = []

# Cleaning Train Data
for line in X_train_raw:
    # Remove handles and hashtags
    line = re.sub("@[A-Za-z0-9_]+","", line)
    line = re.sub("#[A-Za-z0-9_]+","", line)
    
    # Remove links
    line = re.sub(r"http\S+", "", line)
    line = re.sub(r"www.\S+", "", line)
    
    # Remove Punctuation
    words = line.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]

    X_train_clean.append(' '.join(stripped))
    
# Cleaning Test Data
for line in X_test_raw:
    # Remove handles and hashtags
    line = re.sub("@[A-Za-z0-9_]+","", line)
    line = re.sub("#[A-Za-z0-9_]+","", line)
    
    # Remove links
    line = re.sub(r"http\S+", "", line)
    line = re.sub(r"www.\S+", "", line)
    
    # Remove Punctuation
    words = line.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]

    X_test_clean.append(' '.join(stripped))

X_train_cleaned = pd.DataFrame(X_train_clean, columns = ['comments'])
X_train_cleaned.to_csv("Train_clean.csv", index=False)
X_test_cleaned = pd.DataFrame(X_test_clean, columns = ['comments'])
X_test_cleaned.to_csv("Test_clean.csv", index=False)


