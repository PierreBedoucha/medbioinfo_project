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

pred_save = []
true_save = []
pred_prob_save = []

clf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=None, verbose=0)

for i, (train_index, val_index) in enumerate(cv.split(), 1):
    print("Set: ", i)
    print("Training on", len(train_index), "examples")
    print("Testing on", len(val_index), "examples")
    (X_train, X_val) = X[train_index, :], X[val_index, :]
    (Y_train, Y_val) = Y[train_index], Y[val_index]

    clf = clf.fit(X_train, Y_train)

    # Predict on the training data
    pred = clf.predict(X_train)
    # Calculate performance measures on the validation data
    acc_train = accuracy_score(pred, Y_train)
    mcc_train = matthews_corrcoef(pred, Y_train)
    f1_train = f1_score(pred, Y_train)

    # Predict on the validation data
    val_pred = clf.predict(X_val)
    # Predict the probability (to use the roc-plot later)
    val_pred_prob = val_pred

    # Save the values to have predictions for all folds.
    pred_save.append(val_pred)
    pred_prob_save.append(val_pred_prob)
    true_save.append(Y_val)
    # Calculate performance measures on the validation data
    acc = accuracy_score(val_pred, Y_val)
    mcc = matthews_corrcoef(val_pred, Y_val)
    f1 = f1_score(val_pred, Y_val)

    print("Training performance", "f1", f1_train, "acc", acc_train, "mcc", mcc_train)
    print("Validation performance", "f1", f1, "acc", acc, "mcc", mcc)
    print("==============")

# Calculate overall validation performance
predictions = np.concatenate(pred_save)
correct = np.concatenate(true_save)
predicted_prob = np.concatenate(pred_prob_save)
acc = accuracy_score(predictions, correct)
mcc = matthews_corrcoef(predictions, correct)
f1 = f1_score(predictions, correct)
print("==============")
print("Overall Validation Performance", "f1", f1, "acc", acc, "mcc", mcc)
print("==============")

pred_save=np.concatenate(pred_save)
true_save=np.concatenate(true_save)
pred_prob_save=np.concatenate(pred_prob_save)
(fpr, tpr, thres_roc) = roc_curve(true_save, pred_prob_save)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('fpr')
plt.ylabel('tpr')

plt.show()

