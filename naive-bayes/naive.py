#%%
#read dataset
import pandas as pd

data = pd.read_csv('breast-cancer(3).csv')
print(data)

# %%
#search wrong values (?) then drop them (drop column)
clean = data.drop((data.loc[data['node-caps']=='?']).index)
print(clean)

# %%
#check if data still have a wrong values (or '?') if is empty, berarti bener
clean.loc[clean['node-caps']=='?']

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
print(feat)

#%%
#split into training set and test set
from sklearn.model_selection import train_test_split as tr

f_train,f_test,cl_train,cl_test = tr(feat,cl, test_size = 0.3, random_state = 0)
print("training : ",f_train)
print("class : " ,cl_train)

# %%
#Naive Bayes was here (with Gaussian Naive Bayes)
from sklearn.naive_bayes import GaussianNB 

mod = GaussianNB()
mod.fit(f_train,cl_train)
pred = mod.predict(f_test)
print(pred)

# %%
#f1-score count

from sklearn.metrics import f1_score as skor

skor(cl_test,pred,average='macro')

# %%
# accuracy count

from sklearn.metrics import accuracy_score as acc

acc(cl_test,pred)

