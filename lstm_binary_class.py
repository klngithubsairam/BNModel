import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.metrics import cohen_kappa_score
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
# fix random seed for reproducibility
np.random.seed(6)
df=pd.read_csv("/content/gdrive/My Drive/Ph.D/Lstm_Reentrancy_Dataset.csv",delimiter=',')
#df=pd.read_csv("/content/gdrive/My Drive/Ph.D/Lstm_Txo_Dataset.csv",delimiter=',')
#df=pd.read_csv("/content/gdrive/My Drive/Ph.D/Lstm_Dos_Dataset.csv",delimiter=',')

x_train, x_test = train_test_split(df, test_size=0.7,random_state=12)

x1=x_train[['P1','P2','P3','P4','P5','P6']].values
y1 = x_train[['V1']].values
#y1 = x_train[['V2']].values
#y1 = x_train[['V3']].values

#print(x1)
print(x1.shape)
print(y1.shape)

x2=x_test[['P1','P2','P3','P4','P5','P6']].values
y2 = x_test[['V1']].values
#y2 = x_test[['V2']].values
#y2 = x_test[['V3']].values

#print(x2)
print(x2.shape)
print(y2.shape)

# create the model
embedding_vecor_length = 32
model = Sequential()
#model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Embedding(6, embedding_vecor_length, input_length=6))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x1, y1, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(x2, y2, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

yhat_probs = model.predict(x2, verbose=0)
# predict crisp classes for test set
yhat_classes = (model.predict(x2) > 0.5).astype("int32")

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
#accuracy = accuracy_score(y2, yhat_classes)
#print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y2, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y2, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y2, yhat_classes)
print('F1 score: %f' % f1)
matrix = confusion_matrix(y2, yhat_classes)
print("==Confusion Matrix==")
print(matrix)
