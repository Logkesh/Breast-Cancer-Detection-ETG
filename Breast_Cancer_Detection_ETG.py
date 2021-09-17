"""Breast Cancer Detection"""

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

dt = load_breast_cancer()

dt.keys()

print(dt['DESCR'])

values = StandardScaler().fit_transform(dt['data'])

pd.DataFrame(values,columns=dt['feature_names'])

feature = values
df_frt = pd.DataFrame(feature , columns = dt['feature_names'])
df_lbl = pd.DataFrame(dt['target'] , columns = ['label'])
df = pd.concat([df_frt, df_lbl], axis=1)
df = df.sample(frac = 1)

feature = df.values[ : , : 30]
label = df.values[ : ,30: ]

#500 Training
X_train = feature[:500]
y_train = label[:500]

#35 Validation
X_val = feature[500:535]
y_val = label[500:535]

#34 Testing
X_test = feature[535:]
y_test = label[535:]

model = Sequential()

model.add(Dense(32, activation = 'relu', input_dim = 30))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile( loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

model.fit(X_train,y_train,batch_size=5,epochs=10,validation_data=(X_val,y_val))

model.evaluate(X_test,y_test)

j = 0
print("*"*20,"BCD Deep Learning","*"*20)
for i in model.predict(X_val):
  print("Predicted : ",dt['target_names'][round(float(i))])
  print("Actual : ",dt['target_names'][int(y_val[j][0])])
  print("-------------------------")
  j+=1

