#Multi Label Classification with Single Output Layer
#https://stackabuse.com/python-for-nlp-multi-label-text-classification-with-keras/



from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

# load dataset
patterns = pd.read_csv("/content/gdrive/My Drive/Ph.D/lstm_dataset_Multilabel.csv")
print(patterns.shape)
pattern_labels= patterns[["P1", "P2", "P3", "P4", "P5", "P6"]]
output_labels = patterns[["V1", "V2", "V3"]]

X= pattern_labels.values
y = output_labels.values
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train[0:2])
print(y_train[0:2])

maxlen = 6
vocab_size = 6
deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 6, trainable=False)(deep_inputs) #weights=[embedding_matrix],
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(3, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())
history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.3)
score = model.evaluate(X_test, y_test, verbose=1)

#model.fit(X_train, y_train)
#predicted = model.predict(X_test)
#report = classification_report(y_test, predicted)
#print(report)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

--------------------------------------------------------------------------

#Multi-lable Text Classification Model with Multiple Output Layers

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

# load dataset
patterns = pd.read_csv("/content/gdrive/My Drive/Ph.D/lstm_dataset_Multilabel.csv")
print(patterns.shape)
pattern_labels= patterns[["P1", "P2", "P3", "P4", "P5", "P6"]]
output_labels = patterns[["V1", "V2", "V3"]]

X= pattern_labels.values
y = output_labels.values
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train[0:2])
print(y_train[0:2])

y1_train = y_train[:,0]
y1_test =  y_test[:,0]

# Second output
y2_train = y_train[:,1]
y2_test =  y_test[:,1]

# Third output
y3_train = y_train[:,2]
y3_test =  y_test[:,2]

maxlen = 6
vocab_size = 6

input_1 = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 6, trainable=False)(input_1)  #weights=[embedding_matrix], 
LSTM_Layer1 = LSTM(128)(embedding_layer)

output1 = Dense(1, activation='sigmoid')(LSTM_Layer1)
output2 = Dense(1, activation='sigmoid')(LSTM_Layer1)
output3 = Dense(1, activation='sigmoid')(LSTM_Layer1)


model = Model(inputs=input_1, outputs=[output1, output2, output3])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())
history = model.fit(x=X_train, y=[y1_train, y2_train, y3_train], batch_size=8192, epochs=5, verbose=1, validation_split=0.2)

score = model.evaluate(x=X_test, y=[y1_test, y2_test, y3_test], verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])







