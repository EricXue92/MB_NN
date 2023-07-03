import sys
import random 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten,Multiply
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import statistics

tf.random.set_seed(3)
tf.keras.utils.set_random_seed(3)
tf.config.experimental.enable_op_determinism()

class MB_nn:
    def __init__(self, input_shape, num_class, unit = 8,  batch_size = 128, pos_add = [0], 
        domain_knowledge_flag = False, seed = 0):

        self.input_shape = input_shape
        self.num_class = num_class
        self.unit = unit
        self.batch_size = batch_size
        self.pos_add = pos_add
        self.domain_knowledge_flag = domain_knowledge_flag 
        self.seed = seed  
        self.model = self.build_model()

    def build_model(self): 
        inputs = Input(shape = (self.input_shape, ))
        curr = Dense(self.unit, activation = "relu")(inputs)
        curr = Dense(self.unit, activation = "relu")(curr)
        output = Dense(self.num_class, activation = "softmax")(curr)

        return keras.Model(inputs = inputs, outputs = output)
    
    # Return the inject_domain_knowledge loss 
    def Point_wise_Loss(self, x_batch, y_batch):
        if self.domain_knowledge_flag and self.pos_add is not None:
            with tf.GradientTape(persistent=True) as g:
                g.watch(x_batch)
                y = self.model(x_batch)
            # y = np.mean(y, axis = 0)
            # y = tf.convert_to_tensor(y)
            # factor = tf.constant([2.0, 1.0, 1.0, 1.0, 1.0])
            # y_  = y * factor

            dy_dx = g.gradient(y, x_batch)
            #print(dy_dx)
            # non-decreasing monotonicity ( dy_dx >= 0 )
            Loss_mono = 0.0
            # i is the actual position in the input data 
            for i in self.pos_add:
                gradient = dy_dx[:, i]
                # dy_dx < 0
                Loss_mono = Loss_mono + 32 * tf.reduce_mean( tf.maximum(0.0, 1.0 * gradient) )
            
                # dy_dx > 0
                #Loss_mono = Loss_mono + 1000000.0 * tf.reduce_mean( tf.maximum(0.0, -1.0 * gradient) )
            #print(Loss_mono)
            return Loss_mono

        else:
            return 0.0

    def loss_fn(self, x_batch, y_batch):

        y_pred = self.model(x_batch)
        # Computes the crossentropy loss between the labels and predictions.
        cce = tf.keras.losses.CategoricalCrossentropy()
        Loss_NN = cce(y_pred, y_batch)
        
        #Computes the mean of elements across dimensions of a tensor.
        Loss_NN = tf.reduce_mean(Loss_NN)
        print(Loss_NN)

        Loss_mono = self.Point_wise_Loss(x_batch, y_batch)
        total_loss = Loss_NN + Loss_mono 

        return [total_loss] 
    
    def get_grad(self, x_batch, y_batch):
        # Record operations for automatic differentiation.
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(x_batch, y_batch)
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return loss, g  
    
    def assign_data(self, train_X, train_y, val_X, val_y, X_test, y_test):
        self.train_X = train_X.astype('float32')
        self.train_y = train_y.astype('float32')
        # Converts the given value to a Tensor.
        # self.train_X =  tf.convert_to_tensor(self.train_X, dtype=tf.float32)
        # self.train_y =  tf.convert_to_tensor(self.train_y, dtype=tf.float32)
        self.val_X = val_X.astype('float32')
        self.val_y = val_y.astype('float32')
        self.X_test = X_test.astype('float32')
        self.y_test = y_test.astype('float32')
   
    def train(self, epoches, optimizer):

        def train_step(x_batch, y_batch):
            loss, grad_theta = self.get_grad(x_batch, y_batch)
            optimizer.apply_gradients( zip(grad_theta, self.model.trainable_variables) )
            return loss, grad_theta

        # Create a dataset
        train_dataset = tf.data.Dataset.from_tensor_slices( (self.train_X, self.train_y) )
        train_dataset = train_dataset.shuffle(buffer_size = 1024, seed = self.seed).batch(self.batch_size)

        best_value = 1e9
        val_ =[]
        loss_ = []

        for epoch in range(epoches):
            temp = []
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss, grad_theta = train_step( x_batch_train, y_batch_train )
                temp.append(loss[0].numpy())

            loss_.append(statistics.fmean(temp))
            print(loss_)
            y_pred_val = self.model(self.val_X )

            cce = tf.keras.losses.CategoricalCrossentropy()
            val_loss = cce(y_pred_val, self.val_y).numpy()
            val_.append(val_loss)

            if val_loss < best_value: 
                best_value = val_loss
                self.min_val_loss = best_value
                self.best_weights = self.model.get_weights()
                self.count = 0
            else:
                self.count += 1

        plt.plot(val_, label = 'Val_loss')
        plt.plot(loss_, label = 'Train_loss')
        plt.legend()
        plt.show()

