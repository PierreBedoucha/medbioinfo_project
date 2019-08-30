import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train=pd.read_csv('ProQDock.csv')


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

trainable_cols = ["rGb","nBSA","Fintres","Sc","EC","ProQ","Isc","rTs","Erep","Etmr","CPM","Ld", "CPscore"]

train_columns=trainable_cols + ["DockQ-Binary","cv"]
train_data=train[train_columns].dropna()

scaling=True
if scaling:
    columns_to_scale=trainable_cols
    # Fit the scaler on the training data
    min_max_scaler.fit(train_data[columns_to_scale].values)
    # Transform the scaling to the train_data
    train_data.loc[:,columns_to_scale]=min_max_scaler.transform(train_data[columns_to_scale].values)

from sklearn.model_selection import PredefinedSplit
(size_x,size_y)=train_data.shape
target_index=size_y-2
cv_index=size_y-1

X=train_data[trainable_cols].values
Y=train_data['DockQ-Binary'].values

cv = PredefinedSplit(train_data['cv'].values)

import sklearn
import numpy as np
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve

from keras.models import Sequential
from keras.layers import Dense

def plot_loss_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train acc', 'val acc', 'train loss', 'val loss'], loc='upper left')
    plt.show()

model = Sequential()

model.add(Dense(units=3, activation='tanh', input_dim=13))
model.add(Dense(units=3, activation='tanh', input_dim=13))
model.add(Dense(units=3, activation='tanh', input_dim=13))
model.add(Dense(units=3, activation='tanh', input_dim=13))
model.add(Dense(units=2, activation='softmax'))

model.compile(optimizer='rmsprop',                    #adaptive learning rate method
              loss='sparse_categorical_crossentropy', #loss function for classification problems with integer labels
              metrics=['accuracy'])                   #the metric doesn't influence the training

hist = model.fit(X, Y, epochs=10, batch_size=32, validation_split=0.2)

plot_loss_acc(hist)

