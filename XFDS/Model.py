# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:34:30 2021

@author: tahamansouri
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np


# Build the model

def create_base_model(input):
    i = Input(shape=(input,))
    x = Dense(128,activation='relu')(i)
    x = Dense(128,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    y = Dense(128,activation='relu')(x)
    x = Dense(1, activation='sigmoid')(y)
    model = Model(i, x)
    mid_model=Model(i,y)
    return model,mid_model

def create_discriminator_model(input):
    i = Input(shape=(input,))
    x = Dense(128,activation='relu')(i)
    x = Dense(128,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(i, x)
    return model

def create_generator(input, output):
    i = Input(shape=(input,))
    x = Dense(128,activation='relu')(i)
    x = Dense(128,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(output, activation='sigmoid')(x)
    model = Model(i, x)
    return model
    
def my_loss(y_true, y_pred):
    d = y_pred.shape[1]
    sel_prob = y_true[:,:d]
    dis_prob = y_true[:,d:(d+1)]
    val_prob = y_true[:,(d+1):(d+2)]
    y_final = y_true[:,(d+2):]
    Reward1 = tf.reduce_sum(y_final * tf.math.log(dis_prob + 1e-8), axis = 1)
    Reward2 = tf.reduce_sum(y_final * tf.math.log(val_prob + 1e-8), axis = 1)
    Reward = Reward1 - Reward2
    loss1 = Reward * tf.reduce_sum( sel_prob * tf.math.log(y_pred + 1e-8) + (1-sel_prob) * tf.math.log(1-y_pred + 1e-8), axis = 1) - 20 * tf.reduce_mean(y_pred, axis = 1)
    loss = tf.reduce_mean(-loss1)
    return loss

def Sample_M(gen_prob):
    n = gen_prob.shape[0]
    d = gen_prob.shape[1]
    samples = np.random.binomial(1, gen_prob, (n,d))
    return samples

def train(X_train,y_train, batch_size, generator,discriminator,base,itNum,save_model):
    for epoch in range(itNum):
        # Select a random batch of samples
        idx = np.random.randint(0, X_train.shape[0], 64)
        x_batch = X_train[idx,:]
        y_batch = y_train[idx]
        gen_prob = generator.predict(x_batch)
        sel_prob = Sample_M(gen_prob)
        dis_prob = discriminator.predict(x_batch*sel_prob)
        d_loss = discriminator.train_on_batch(x_batch*sel_prob, y_batch)
        val_prob = base.predict(x_batch)
        #v_loss = base.train_on_batch(x_batch, y_batch)
        y_batch_final = np.concatenate( (sel_prob, np.asarray(dis_prob), np.asarray(val_prob), y_batch.reshape(-1,1)), axis = 1 )
        g_loss = generator.train_on_batch(x_batch, y_batch_final)
        dialog = 'Epoch: ' + str(epoch) + ', d_loss (Acc)): ' + str(d_loss[1]) + ', g_loss: ' + str(np.round(g_loss,4))
        print(dialog)
        if d_loss[1]==1.0:
            break
    if save_model:
        print('All models are saved!!!')
        generator.save('generator.ml')
        discriminator.save('discriminator.ml')
        base.save('base.ml')
    print('End of training!!!')
    return generator,discriminator,base
    
    


  
  




