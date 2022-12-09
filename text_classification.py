#%% 
import pandas as pd
import os,re, datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

#%% Step 1) Data Loading
csv_path = os.path.join(os.getcwd(), 'Dataset', 'True.csv')
df = pd.read_csv(csv_path)

text = df['text'] # features
subject = df['subject'] # target

#%% Step 2) Data Inspection
df.head()

#%%
df.isna().sum()

#%%
df.info()

#%%
df.describe()

#%%
df.duplicated().sum()
#%%
# pick any text
print(text[10])

#%%
temp = [] # to check how many bit.ly in the text
for s in text:
    if 'bit.ly/' in s:
        temp.append(s)
#%% Step 3) Data Cleaning

temp = []
for index, txt in enumerate(text):
    text[index] = re.sub(r'(^[^-]*)|(@[^\s]+)|bit.ly/\d\w{1,10}|(\s+EST)|[^a-zA-Z]', ' ', txt).lower()
    temp.append(len(text[index].split())) # to check the number of words of all the sentences

#%%
df1 = pd.concat([text,subject], axis=1)
df1 = df1.drop_duplicates()

text = df1['text']
subject = df1['subject']

#%%
print(text[10])
print(np.mean(temp)) # to find the average number of words
#%% Step 4) Features Selection
#%% Step 5) Data Pre-processing
# for features
# use tokenizers
num_words = 5000 
oov_token = 'OOV'

tokenizer = Tokenizer(num_words=num_words, oov_token = oov_token)
tokenizer.fit_on_texts(text)

word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(text)

#%%
# use pad and trunctuating
# to convert text to horizontal instead of vertical
# to convert a word into number limit to 200 words per sentence
np.mean(temp)
train_sequences = pad_sequences(train_sequences, maxlen=400, padding='post', truncating='post' )

#%%
# for target
print(subject.unique)

# convert target into numerical, politicsNews =  1.0 and worldnews = 0.1
ohe = OneHotEncoder(sparse=False)
train_subject = ohe.fit_transform(subject[::, None])


#%% Model Development
# train test split
train_sequences = np.expand_dims(train_sequences, -1)

X_train, X_test, y_train, y_test = train_test_split(train_sequences, train_subject)
#%% Model Evaluation
embedding_size = 64
model = Sequential()

#Bidirectional us for looking missing words
# use embedding as input layer
model.add(Embedding(num_words, embedding_size)) # num_words = 5000
model.add(LSTM(embedding_size,return_sequences = True))
model.add(LSTM(embedding_size))
model.add(Dense(2, activation = 'softmax')) # 2 outputs
model.summary()

#%% Model Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')  # loss cannot be used sparse bc this is one hot encoder. 

#%%
logs_path = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = TensorBoard(log_dir=logs_path)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, epochs=5, callbacks=[tensorboard_callback, early_stop_callback], validation_data=(X_test, y_test))

#%% Model Evaluation
y_pred = model.predict(X_test)

# %%
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test,axis=1)

print(classification_report(y_true, y_pred))
 #%% Model Saving
# save tokenizer (txt --> number)

import json

token_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(token_json, f)

# %% save ohe

import pickle
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe,f)

#%%

model.save('model sentiment analysis.h5')
# %%
