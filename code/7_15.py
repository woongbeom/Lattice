import os, glob, numpy as np
#from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dropout, AveragePooling1D, AveragePooling2D, Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
#import keras.backend.tensorflow_backend as K
from keras import backend as K

from tensorflow.python.keras.layers import Input, Dense, Embedding, ZeroPadding2D
from tensorflow.python.keras.models import Sequential
from keras.optimizers import SGD

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

X_train, X_test, Y_train, Y_test = np.load('H:/1. Research/3. PoDoc_Kim/1. CoCr_AI design/2. Python Code/n_comp_n_hatch_CODE_NPY/7_15.npy', allow_pickle = True)
print(X_train.shape)
print(X_train.shape[0])




X_train = X_train.reshape(X_train.shape[0], 7, 1)
X_test = X_test.reshape(X_test.shape[0], 7, 1)



# In[ ]:


categories = ["Feasible", "Infeasible" ]
nb_classes = len(categories)


# In[ ]:

model = Sequential()

model.add(Conv1D(32, 3, padding="same", input_shape=(7, 1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv1D(64, 3, padding="same", activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

# model.add(AveragePooling1D(pool_size = 2, strides=None, padding="valid", data_format=None))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))


model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=[('accuracy')])

# model.compile(loss='categorical_crossentropy', optimizer='adam')

# model.add(LSTM(20, input_shape=(7, 1))) # (timestep, feature) 
# model.add(Dense(2)) # output = 1 
# model.compile(loss='mean_squared_error', optimizer='adam')




model_dir = 'H:/1. Research/3. PoDoc_Kim/1. CoCr_AI design/2. Python Code/model'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_path = model_dir + '/doe_data'
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=6)



# In[ ]:


model.summary()


# In[ ]:
    
# X_train = tf.squeeze(X_train, axis=-1)

    
tf.random.set_seed(0)

import time

start = time.time()

# with tf.device('GPU:0'):
    # history = model.fit(X_train_t, Y_train, batch_size=7, epochs=30, validation_data=(X_test, Y_test), callbacks=[checkpoint, early_stopping])

early_stop = EarlyStopping(monitor='loss', patience=20, verbose =1)
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=30, verbose=1, callbacks=[early_stop])

# In[ ]:


print("정확도 : %.4f" %(model.evaluate(X_test, Y_test)[1] * 100), "%")

print("학습 소요 시간 :", round(time.time()-start, 2), "초")



from keras.models import load_model

model.save('H:/1. Research/3. PoDoc_Kim/1. CoCr_AI design/2. Python Code/model/7_15.h5', overwrite=True)

# Confusion matrix

from sklearn.metrics import confusion_matrix #교차표 그리기   

prediction = model.predict(X_test)

cfm=confusion_matrix(Y_test.argmax(axis=1), prediction.argmax(axis=1))

import seaborn as sns
sns.set(font_scale=2.0)
ax = sns.heatmap(cfm, annot=True, cmap='Blues', fmt='g')

ax.set_title('7_comp_15_hatching');

## Display the visualization of the Confusion Matrix.
plt.show()


# In[ ]:


y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()


# In[10]:


# 10000개 학습 시간 6.28초
# 1개 소요시간 0.0226초
# 8 ^ 7 =  2,097,152 ea  <-- 가능한 모든 lattice 순서의 경우의 수
#  2,097,152 ea * 0.025 s = 52,429 s = 14.56 h

# Optistruct 해석 1개 해석 소요시간 8초
#  2,097,152 ea * 8 s =   16,777,216 s = 4,660 j = 194.18 day














