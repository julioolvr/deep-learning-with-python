from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))

  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.

  return results

word_index = imdb.get_word_index()
some_review = 'the movie sucks everything is terrible i would not watch it again horrible awful'
encoded_review = [1] + map(lambda s: word_index.get(s, -1) + 3, some_review.split(' '))

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model.fit(partial_x_train,
          partial_y_train,
          epochs=4,
          batch_size=512,
          validation_data=(x_val, y_val))

print model.predict(vectorize_sequences([encoded_review]))