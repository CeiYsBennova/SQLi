import numpy as np
import pandas as pd

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
train_data, test_data = data[:40000], data[40000:]

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

# Train the model
num_epochs = 10
history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels), verbose=2)

# save the model
#model.save('sqlidl.h5')
#predict the model
sql = ["SELECT id, key, helo  FROM abc  WHERE id  IN  ( 3 )  ORDER BY id ASC "]
sql_sequence = tokenizer.texts_to_sequences(sql)
sql_padded = pad_sequences(sql_sequence, maxlen=max_length, padding='post', truncating='post')

if model.predict(sql_padded)[0][0] > 0.5:
    print("SQLi")
else:
    print("Normal")




