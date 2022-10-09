#prueba sacando add
from tabnanny import verbose
from model import Unet
import matplotlib.pyplot as plt

import numpy as np

#import os
#import random
import numpy as np

#from tqdm import tqdm_notebook, tnrange
#from itertools import chain

import tensorflow as tf

#from keras.models import Model, load_model
#from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
##from tensorflow.keras.layers.core import Lambda, RepeatVector, Reshape
#from keras.layers.convolutional import Conv2D, Conv2DTranspose
#from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
#from keras.layers import concatenate, add
#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
#from tensorflow.keras.layers.core import Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
#from tensorflow.keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
#from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

class ConvUnet(Unet):
    
    def __init__(self):
        super().__init__()
        self.name="Unet convencional (Ronneberger)"
        input_img = Input((self.rows, self.cols, 1), name='img')
        self.get_unet(input_img)

    def conv2d_block(self, input_tensor, n_filters, kernel_size = 3, batchnorm = True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # second layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                kernel_initializer = 'he_normal', padding = 'same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
  
    def get_unet(self, input_img, n_filters = 16, dropout = 0.2, batchnorm = True):
        # Contracting Path
        c1 = self.conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)
        
        c2 = self.conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)
        
        c3 = self.conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)
        
        c4 = self.conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)
        
        c5 = self.conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
        
        # Expansive Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        
        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        
        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        
        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        self.model = Model(inputs=[input_img], outputs=[outputs])
    
    def ModelCompile(self):
        self.model.compile(optimizer = Adam(),loss="binary_crossentropy",metrics = ["accuracy"])#,run_eagerly=True)

    def ModelFit(self,X,Y,X_valid,y_valid):
        callbacks = [
            EarlyStopping(monitor="val_loss", min_delta=0,patience=10,verbose=1), #puedo probar con otro delta
            ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=0.0001,verbose=1),
            ModelCheckpoint('BestModel.h5',verbose=1,save_best_only=True,save_weights_only=True)
            ]
        history=self.model.fit(X,Y, batch_size=1, epochs=200, validation_data=(X_valid,y_valid), callbacks=callbacks)#epochs=200
        
        plt.plot(history.history["loss"],label="loss")
        plt.plot(history.history["val_loss"],label="val_loss")
        plt.plot(np.argmin(history.history["val_loss"]),np.min(history.history["val_loss"]),marker="x",color="r",label="best model")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Learning curve")
        plt.savefig("learning_history.png")
    
    def ModelPredict(self,X,device):
        '''Return predictions over the best model obtained during early stopping
            X: test set
            device: device used for inference. Options: "gpu" or "cpu"
        '''
        self.model.load_weights("D:/Proyectos ML/Brain tumor/.venv/Scripts/BestModel.h5")
        with tf.device(f'/{device}:0'):
            return self.model.predict(X)
            

    def print_examples(self,y_test,post_pred,index={}):
        '''Prints predictions'''
        super().print_examples(y_test,post_pred,index,"ConvUnet")

            


