#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
import sys
import gym
import zipfile
import autograd
import matplotlib.gridspec as gridspec
# Use tf.random.set_seed for TensorFlow 2.0 and above
#from scipy.signal.waveforms import square
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import Input
from tensorflow.keras import layers


# In[15]:


# @title  Hp meristor's state variable:
from IPython.display import display, Math

latex_equation = r"""
\text{State variable:}\quad \frac{dw}{dt} = \mu_\text{v} \cdot \left( \frac{R_{\text{on}}}{D^2} \right) \cdot i(t) \cdot f(w) \\
\text{Window function:}\quad f(w) = {w(1 - w)}\\
\text{state variable in this code is w:}\quad w =  \frac{X}{D}
"""
display(Math(latex_equation))


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------- Physical Parameters ----------------
frequency = 1
A_train = 1.5
W_train = 2 * np.pi * frequency

mu_v = 10**4
D = 60  
r_on = 0.1  
r_off = 16  
r0 = 4
w0 = (r0 - r_off) / (r_on - r_off)  

points_per_period = 600  

total_points = 10 * points_per_period  

# -------------- Solving ODE ----------------
def f(t, w, A, W, mu_v, D, r_on, r_off):
    k = mu_v * (r_on / D**2)
    f_w = w * (1 - w)
    r = r_on * w + r_off * (1 - w)
    I = A * np.sin(W * t) / r
    return I * f_w * k

t_all = np.linspace(0, 6, total_points)  
sol_all=solve_ivp(f, (0, 6), [w0], t_eval=t_all, args=(A_train, W_train, mu_v, D, r_on, r_off),
          method='RK45', max_step=0.001)


w_all = sol_all.y[0]
v_all = A_train * np.sin(W_train * t_all)
r_all = r_on * w_all + r_off * (1 - w_all)
I_all = v_all / r_all

X_all = np.column_stack([t_all[:-1], w_all[:-1], I_all[:-1]])  
y_all = w_all[1:]  

# -------------- Split Data ----------------
test_ratio = 0.2  
test_size = int(test_ratio * len(X_all))


test_index = np.arange(len(X_all) - test_size, len(X_all))
train_index = np.arange(0, len(X_all) - test_size)


X_train, X_test = X_all[train_index], X_all[test_index]
y_train, y_test = y_all[train_index], y_all[test_index]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
plt.figure(figsize=(10, 4))
plt.plot(t_all[:-1], w_all[:-1], label="Original Data", alpha=0.3)
plt.plot(X_train[:, 0], y_train, color='blue', alpha=0.5, label='Train')
plt.plot(X_test[:, 0], y_test, color='red', alpha=0.5, label='Test')
plt.xlabel("Time")
plt.ylabel("w (State Variable)")
plt.title("Train/Test Split for Time-Series Memristor Data")
#plt.grid()
plt.legend()
plt.savefig("rungkutta_train_test.pdf") 
plt.show()

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# ------------------ نرمال‌سازی ------------------
scaler_time = StandardScaler()
X_train[:, 0] = scaler_time.fit_transform(X_train[:, 0].reshape(-1, 1)).flatten()
X_test[:, 0] = scaler_time.transform(X_test[:, 0].reshape(-1, 1)).flatten()  

scaler_features = MinMaxScaler(feature_range=(-1, 1))
X_train[:, 1:] = scaler_features.fit_transform(X_train[:, 1:])
X_test[:, 1:] = scaler_features.transform(X_test[:, 1:])  

scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))


X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()



# ---------------- Create Sequences ----------------
sequence_length = 10  # 
def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])  
    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)  



X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)



print("Mean of X_train_scaled:", np.mean(X_train, axis=0))
print("Std of X_train_scaled:", np.std(X_train, axis=0))
print("Mean of X_test_scaled:", np.mean(X_test, axis=0))
print("Std of X_test_scaled:", np.std(X_test, axis=0))
print("Mean of y_train_scaled:", np.mean(y_train_scaled))
print("Std of y_train_scaled:", np.std(y_train_scaled))
print("Mean of y_test_scaled:", np.mean(y_test_scaled))
print("Std of y_test_scaled:", np.std(y_test_scaled))



print(f'X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}')
print(f'X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}')


# In[9]:


import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM # Import LSTM here
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dropout

class RNN(tf.keras.Model):
    def __init__(self, **kwargs):  
        super().__init__(**kwargs)  
        self.RNN = Sequential([
            LSTM(35, return_sequences=True, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            Dropout(0.06),
            LSTM(39, return_sequences=True, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),  # تغییر این خط
            Dropout(0.06),
            Dense(29, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            Dense(1,)  
        ])
    

    def call(self, inputs):
        return self.RNN(inputs)

    def build(self, input_shape):
        self.RNN.build(input_shape)
        super().build(input_shape)


rnn = RNN()
rnn.build((None, sequence_length, 3))  
rnn.summary()
####################################################################
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt

# --------


# In[10]:


N0 = 1  
Nf = X_train_seq.shape[0]  
Nd = y_train_seq.shape[0]

#col_weights = tf.Variable(1.0)   # for ode loss
#u_weights = tf.Variable(1.0)     # for ic loss
#data_weights = tf.Variable(1.5)  # for data loss
#________________________________________
#col_weights = tf.Variable(tf.ones(Nf), dtype=tf.float32)   #weight of ODE
data_weights = tf.Variable(tf.ones(Nd), dtype=tf.float32)  #weight of data
u_weights = tf.Variable(tf.ones(N0), dtype=tf.float32)    ##weight of IC     

#optimizer_col_weights = tf.keras.optimizers.Adam(learning_rate=1e-2)
optimizer_data_weights = tf.keras.optimizers.Adam(learning_rate=1e-4) 

#print("Shape of col_weights:", col_weights.shape)
print("Shape of ode_res:", data_weights .shape)

print("done")


# In[11]:


def compute_loss(X, y_true, mode, u_weights, data_weights):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    
    with tf.GradientTape(persistent=True) as tape:
    
        w_pred_sequence = model(X)
        w_pred = w_pred_sequence[:, -1, :]

    
        I_t = X[:, -1, 0:1]
        w_prev = X[:, -1, 1:2]
        T = X[:, -1, 2:3]

        f_w = w_pred * (1 - w_pred)

        with tf.GradientTape() as g:
            g.watch(T)

            inputs = tf.concat([I_t,w_prev, T], axis=1)
            w_pred_g = model(tf.expand_dims(inputs, axis=1))  # (batch_size, 1, 3)
        dw_dt = g.gradient(w_pred_g, T)

        #ode_res = dw_dt - mu_v * (r_on / D**2) * I_t * f_w
      
        #ode_loss = tf.reduce_mean(tf.square(col_weights[:, tf.newaxis] * ode_res))

        data_loss = tf.reduce_mean(tf.square(data_weights * (w_pred - y_true)))
        
        
        #ic_input = tf.convert_to_tensor([[0.0, y_train[0], X_train[0, 2]]], dtype=tf.float32)
        ic_input = X[:, 0:1, :]
        ic_pred = model(ic_input)[:, -1, :]
        
        ic_true = tf.convert_to_tensor(y_train[0], dtype=tf.float32)
        ic_loss = tf.reduce_mean(tf.square(u_weights * (ic_pred - ic_true)))

        total_loss = data_loss + ic_loss

        return total_loss, data_loss, ic_loss


# In[14]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------- Learning Rate Schedule ----------------
from tensorflow.keras.optimizers.schedules import ExponentialDecay
#earning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
   #initial_learning_rate=1e-3  
    #ecay_steps=10000 ,
    #ecay_rate=0.75
#
learning_rate=1e-3



optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0)  

# ---------------- ModelCheckpoint Callback ----------------
checkpoint = ModelCheckpoint(
    'best_model.keras',  
    save_best_only=True,
    monitor='val_loss'
)

# ---------------- Compile the Model ----------------
rnn.compile(
    optimizer=optimizer,
    loss='mean_squared_error',
    metrics=['mae']
)

# ---------------- Training Loop ----------------
history =rnn.fit(
    X_train_seq, 
    y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=700,
    batch_size=32,
    callbacks=[checkpoint]  
)


# In[13]:


w_pred_seq = rnn.predict(X_test_seq)  
w_pred = w_pred_seq[:, -1, :]  

w_pred_original = scaler_y.inverse_transform(w_pred)  


y_test_original = scaler_y.inverse_transform(y_test_seq)


t_test_plot = t_all[len(t_all) - len(w_pred_original):]  


# --------------- Plotting ----------------
plt.figure(figsize=(10, 6))
plt.plot(t_test_plot, y_test_original, label='True', linewidth=2)
plt.plot(t_test_plot, w_pred_original, label='Predicted', linestyle='--', linewidth=4)
plt.xlabel('Time')
plt.ylabel('w(t)')
plt.title('LSTM Prediction vs True State Variable')
plt.savefig("testlstm.pdf")
plt.legend()

plt.show()


# In[ ]:





# In[ ]:




