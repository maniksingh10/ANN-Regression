import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder

df = pd.read_csv('data.csv')
df.head()

df.drop(['RowNumber','CustomerId','Surname'],axis=1, inplace=True)
df.dropna(inplace=True)

le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
oh = OneHotEncoder(sparse_output = False)
done_oh = oh.fit_transform(df[['Geography']])
oh.get_feature_names_out(['Geography'])

pd.DataFrame(done_oh, columns=oh.get_feature_names_out(['Geography']))

data = pd.concat([df.drop(['Geography'],axis=1), pd.DataFrame(done_oh, columns=oh.get_feature_names_out(['Geography']))], axis=1)

data.head()

with open('gen_encoder.pkl', 'wb') as f:
    pickle.dump(le_gender, f)

with open('loc_encoder.pkl', 'wb') as f:
    pickle.dump(oh, f)


X = data.drop(['EstimatedSalary'], axis=1)
y = data['EstimatedSalary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64,activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32,activation='relu'),
    Dense(1)
])


model.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mae"])

model.summary()

import datetime
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

log_dir = "rlogs/" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
tf_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history=model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[tf_callback,early_stopping]
)
model.save('r_model.h5')

