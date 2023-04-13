import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load tokenizer
tokenizer = Tokenizer()
max_length = 120

model = load_model('sqlidl.h5')

# input string from keyboard to list
string = input("Enter a string: ")
string = [string]

# convert string to sequence
sequences = tokenizer.texts_to_sequences(string)
sql_padded = pad_sequences(sequences, maxlen=max_length, truncating='post')

# predict
if model.predict(sql_padded)[0][0] > 0.5:
    print("SQLi")
else:
    print("Normal")