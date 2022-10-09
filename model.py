#prueba sin add
from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt

import h5py

from sklearn.model_selection import train_test_split
from skimage.transform import resize



class Unet:
    
    def __init__(self):
        self.rows=512
        self.cols=512
    
    def load_data(self):
        data_folder="D:/Proyectos ML/Brain tumor/brain tumor/"
        cant_img=766
        imgs=np.zeros((cant_img,self.rows,self.cols))
        masks=np.zeros((cant_img,self.rows,self.cols))
        for i in range(1,cant_img+1):
            f = h5py.File(data_folder+f'{i}.mat','r')
            imgs[i-1,:,:] =   resize(np.array(f['cjdata'].get("image")), (self.rows, self.cols), mode = 'constant', preserve_range = True)
            masks[i-1,:,:]  =  resize(np.array(f['cjdata'].get("tumorMask")), (self.rows, self.cols), mode = 'constant', preserve_range = True)
        X_train,X_test,y_train,y_test = train_test_split(imgs,masks,test_size=0.1,random_state=42)
        X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.1,random_state=42)
        #return X_train,y_train,X_valid,y_valid,X_test,y_test
        return X_train[1:,:,:],y_train[1:,:,:],X_valid,y_valid,X_test,y_test

    def compile(self):
        pass
    def fit(self):
        pass
    def predict(self):
        pass
    
    def diceScore(self,im_true, im_pred):
        '''Calcula el score de dice como 2*intersección/union'''
        intersection = np.sum(im_true*im_pred)
        sum_areas=np.sum(im_true)+np.sum(im_pred)
        return 2*intersection/sum_areas
    
    def diceMetrics(self,test_mask,predictions,boxplot=""):
        '''return list with dice score for each test image'''
        dice = []
        n=len(test_mask)
        for k in range(0,n):
            im_true=test_mask[k,:,:]
            im_pred=predictions[k,:,:]
            im_pred=im_pred[:,:,0]
            im_pred[im_pred<0.5]=0
            dice.append(self.diceScore(im_true,im_pred))
        if(boxplot!=""):
            plt.boxplot(dice)
            plt.savefig(f"{boxplot}")
            plt.close()
        return dice
    
    def post_processing(self,predictions):
        '''Performs binarizations of the resulting masks'''
        #Tendria que agregar una limpieza morfológica
        predictions[predictions>0.5]=1
        return predictions
    
    def print_examples(self,X_test,post_pred,name,index={}):
        '''Prints predictions'''
        #No funciona
        #Debería agregar la máscara traslúcida
        for idx in index:
            f, ax = plt.subplots(1, 1, sharey=True)
            im=X_test[idx,:,:]
            ax.imshow(im)
            plt.savefig(f"{name}_{idx}.png")