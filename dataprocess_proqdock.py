import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train=pd.read_csv('ProQDock.csv')

min_cat = np.min(train.loc[(train['DockQ-Binary'] == 1)]['DockQ']) + np.std(train['DockQ'])
max_cat = np.max(train.loc[(train['DockQ-Binary'] == 1)]['DockQ']) - np.std(train['DockQ'])

# v = train.loc[(train['DockQ-Binary'] == 1)]['DockQ'].values
v = train['DockQ'].values
cats = np.array(['danger', 'warning', 'success'])
code = np.searchsorted([0.3, 0.8], v.ravel()).reshape(v.shape)
train['DockQ_cat'] = cats[code]
print(train)