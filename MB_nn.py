#https://stackoverflow.com/questions/61550026/valueerror-shapes-none-1-and-none-3-are-incompatible
import sys
import random 
import tensorflow as tf
import keras
from keras import regularizers
from keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report

tf.random.set_seed(2)
tf.keras.utils.set_random_seed(2)
tf.config.experimental.enable_op_determinism()


class MB_nn:
    def __init__(self, input_shape, num_class, batch_size = 256, pos_add = [1,2,3,5], 
        domain_knowledge_flag = False, seed = 0):

        self.input_shape = input_shape
        self.num_class = num_class
        self.batch_size = batch_size
        self.pos_add = pos_add
        self.domain_knowledge_flag = domain_knowledge_flag 
        self.seed = seed  
        self.model = self.build_model()

    def build_model(self): 
        inputs = Input(shape = (self.input_shape, ))
        curr = Dense(16, activation = "relu")(inputs)
        curr = Dense(16, activation = "relu")(curr)
        output = Dense(self.num_class, activation = "softmax")(curr)
        return keras.Model(inputs = inputs, outputs = output)
    
    # Return the inject_domain_knowledge loss 
    def Point_wise_Loss(self, x_batch):
        if self.domain_knowledge_flag and self.pos_add is not None:
            with tf.GradientTape(persistent=True) as g:
                g.watch(x_batch)
                y = self.model(x_batch)
            dy_dx = g.gradient(y, x_batch)
            # non-decreasing monotonicity ( dy_dx >= 0 )
            Loss_mono = 0.0
            # i is the actual position in the input data 
            for i in self.pos_add:
                gradient = dy_dx[:, i]
                Loss_mono = Loss_mono + tf.reduce_mean(tf.maximum(0.0, -1.0 * gradient))
            print(Loss_mono)
            return Loss_mono

        else:
            return 0.0
    
    def loss_fn(self, x_batch, y_batch):
        y_pred = self.model(x_batch)
        cce = tf.keras.losses.CategoricalCrossentropy()
        Loss_NN = cce(y_pred, y_batch)
        # Get the injected loss 
        Loss_mono = self.Point_wise_Loss(x_batch)
        total_loss = Loss_NN + Loss_mono
        return [total_loss] 
    
    def get_grad(self, x_batch, y_batch):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(x_batch, y_batch)
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return loss, g  
    
    def assign_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
    
        self.X_train = X_train.astype('float32')
        self.y_train = y_train.astype('float32')
        self.X_val = X_val.astype('float32')
        self.y_val = y_val.astype('float32')
        self.X_test = X_test.astype('float32')
        self.y_test = y_test.astype('float32')
   
    def train(self, epoches, optimizer):

        @tf.function 
        def train_step(x_batch, y_batch):
            loss, grad_theta = self.get_grad(x_batch, y_batch)
            # Perform Gradient Descent step ( Update weights )
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss, grad_theta

#https://stackoverflow.com/questions/46444018
#/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

        # Create a dataset（constructing many x corresponding to y )
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_dataset = train_dataset.shuffle(buffer_size = 1024, seed = self.seed).batch(self.batch_size)

        best_value = 1e9
        for epoch in range(epoches):
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss, grad_theta = train_step(x_batch_train, y_batch_train)

            y_pred_val = self.model(self.X_val)
            
            cce = tf.keras.losses.CategoricalCrossentropy()
            val_loss = cce(y_pred_val, self.y_val).numpy()
     
            if val_loss < best_value:

                best_value = val_loss
                self.min_val_loss = best_value
                self.best_weights = self.model.get_weights()
                #print("Saving model")
                self.count = 0
            else:
                self.count += 1