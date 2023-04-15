import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

import tensorflow as tf

# import word embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam 

# read csv file
df = pd.read_csv('SQLi.csv')

#df = df[df['Label'].isin(['0', '1'])]

# convert to numpy array
data = df.values

# split into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# split into sentences and labels
train_sentences, train_labels = train_data[:, 0], train_data[:, 1]
test_sentences, test_labels = test_data[:, 0], test_data[:, 1]

train_labels = train_labels.astype(np.float64)
test_labels = test_labels.astype(np.float64)

# create a vocabulary
vocab_size = 10000
embedding_dim = 16
max_length = 120

# create a tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_sentences)

# save the tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# convert the sentences to index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

# pad the sequences
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# create the model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(GlobalAveragePooling1D())
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#call back model
callback = tf.keras.callbacks.ModelCheckpoint('sqlidl.h5', monitor='val_loss')
# Train the model
num_epochs = 10
history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels), verbose=2, callbacks=[callback])

#predict the model
def predict(sentence):
    sequences = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    if model.predict(padded)[0][0] > 0.5:
        print('SQLi')
    else:
        print('Normal')

# predict the model
predict('SELECT Employees.LastName, COUNT ( Orders.OrderID )  AS NumberOfOrders FROM  ( Orders INNER JOIN Employees ON Orders.EmployeeID  =  Employees.EmployeeID )  GROUP BY LastName HAVING COUNT ( Orders.OrderID )  > 10;')
predict('SELECT * FROM users WHERE username = "admin" AND password = "password" OR 1=1')
predict('Hello World!')
predict('union select version(),user(),3,4,--+-')
predict('from users where id  =  1<@<@ union select 1,version()-- 1')
predict('SELECT min (failed) FROM nation SELECT SUM(economy)')
predict('UnIOn sElecT 1,2,3,id(),--+-')

# check weights and accuracy
print(model.evaluate(test_padded, test_labels))






