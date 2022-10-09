#prueba sacando add
from conventional_unet import ConvUnet
from matplotlib import pyplot as plt
import tensorflow as tf
#from tensorflow import keras

mod=ConvUnet()
X_train,y_train,X_valid,y_valid,X_test,y_test = mod.load_data() #(imgs,rows,cols)
 
with tf.device('/cpu:0'):    
    mod.ModelCompile()
    n_train=300
    mod.ModelFit(X_train[0:n_train,:,:],y_train[0:n_train,:,:],X_valid,y_valid) #No entran todas las imágenes en memoria

n_test=3#len(X_test)
predictions = mod.ModelPredict(X_test[0:n_test,:,:],"cpu")
post_pred = mod.post_processing(predictions)
mod.diceMetrics(y_test[0:n_test,:,:],post_pred,"boxplot_05")
#mod.print_examples(X_test,post_pred,{0,10,15})


f, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(X_test[0,:,:])
ax2.imshow(post_pred[0,:,:])
plt.show()

f, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(X_test[1,:,:])
ax2.imshow(post_pred[1,:,:])
plt.show()

f, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(X_test[2,:,:])
ax2.imshow(post_pred[2,:,:])
plt.show()
