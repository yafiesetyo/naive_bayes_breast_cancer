#%%
#read dataset
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB 

data = pd.read_csv('breast-cancer(3).csv')
data

# %%
#search wrong values (?) then drop them (drop column)
target = data.loc[(data['breast-quad']=='?') | (data['node-caps']=='?')]
clean = data.drop(target.index)
print(clean)

# %%
#convert to numerical from string datas (kayaknya salahnya disini)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

c = le.fit_transform(clean['class'])
age = le.fit_transform(clean['age'])
menopause = le.fit_transform(clean['menopause'])
tumor_size = le.fit_transform(clean['tumor-size'])
inv_nodes = le.fit_transform(clean['inv-nodes'])
node_caps = le.fit_transform(clean['node-caps'])
deg_malig = le.fit_transform(clean['deg-malig'])
breast = le.fit_transform(clean['breast'])
breast_quad = le.fit_transform(clean['breast-quad'])
irradiat = le.fit_transform(clean['irradiat'])

cl = c.tolist()
features=zip(age,menopause,tumor_size,inv_nodes,node_caps,deg_malig,breast,breast_quad,irradiat)
feat = list(features)
print(cl)
print(feat)

#%%
#split into training set,validation set, and test set
from sklearn.model_selection import train_test_split as tr
import math

f_train,f_test,cl_train,cl_test = tr(feat,cl, test_size = 0.3, random_state = 42)
l2 = len(f_train)*0.8
length = math.ceil(l2)
f_val,cl_val,fn_train,cln_train = [],[],[],[]
f_val = f_train[length:]
cl_val = cl_train[length:]
fn_train = f_train[0:length]
cln_train = cl_train[0:length]
print('class validation',len(cl_val))
print('att validation : ',len(f_val))
print('att train : ',len(fn_train))
print('class train : ',len(cln_train))
# %%
#Naive Bayes was here (with Gaussian Naive Bayes)

mod = GaussianNB()
mod.fit(fn_train,cln_train)
pred = mod.predict(f_val)
pred

# %%
#f1-score count

from sklearn.metrics import f1_score as skor

skor(cl_val,pred,average='macro')

# %%
# accuracy count

from sklearn.metrics import accuracy_score as acc

acc(cl_val,pred)


# %%
#now for the truth... we test that model with test set

pred_real = mod.predict(f_test)
acc(cl_test,pred_real)
skor(cl_test,pred_real,average='macro')







# %%
