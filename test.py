from conventional_unet import ConvUnet
from matplotlib import pyplot as plt
from tensorflow import keras

mod=ConvUnet()
X_train,y_train,X_valid,y_valid,X_test,y_test = mod.load_data() #(imgs,rows,cols)

#mod.ModelCompile()
#mod.ModelFit(X_train[1:300,:,:],y_train[1:300,:,:],X_valid[:,:,:],y_valid[:,:,:])
#mod.model.save("D:/Proyectos ML/Brain tumor/brain tumor/full.h5")

new_model=keras.models.load_model("D:/Proyectos ML/Brain tumor/brain tumor/full.h5")
#predictions=mod.ModelPredict(X_test[:,:,:])
predictions = new_model.predict(X_test)

dice = mod.diceMetrics(y_test,predictions)
plt.boxplot(dice)
plt.savefig("full_boxplot.png")

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(X_test[0,:,:])
ax2.imshow(y_test[0,:,:])
pred=predictions[0,:,:]
pred[pred<0.5]=0
ax3.imshow(pred)
plt.show()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(X_test[10,:,:])
ax2.imshow(y_test[10,:,:])
pred=predictions[10,:,:]
pred[pred<0.5]=0
ax3.imshow(pred)
plt.show()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(X_test[15,:,:])
ax2.imshow(y_test[15,:,:])
pred=predictions[15,:,:]
pred[pred<0.5]=0
ax3.imshow(pred)
plt.show()