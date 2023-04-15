import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('SQLi.csv')

#df = df[df['Label'].isin(['0', '1'])]

# convert to numpy array
data = df.values

# split into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

test_sentences, test_labels = test_data[:, 0], test_data[:, 1]

test_labels = test_labels.astype(np.float64)

# create a vocabulary
vocab_size = 10000
embedding_dim = 16
max_length = 120

#load tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
# create a tokenizer
test_sequences = tokenizer.texts_to_sequences(test_sentences)

# pad the sequences
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')


model = load_model('sqlidl.h5')

print(model.evaluate(test_padded, test_labels))


# input string from keyboard to list
def predict(sentence):
    sequences = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    if model.predict(padded)[0][0] > 0.5:
        print('SQLi')
    else:
        print('Normal')

string = str(input("Enter a string: "))
print(string)
predict(string)