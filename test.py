from conventional_unet import ConvUnet
from matplotlib import pyplot as plt
from tensorflow import keras

mod=ConvUnet()
X_train,y_train,X_valid,y_valid,X_test,y_test = mod.load_data() #(imgs,rows,cols)

#mod.ModelCompile()
#mod.ModelFit(X_train[0:300,:,:],y_train[0:300,:,:],X_valid,y_valid) #No entran todas las imágenes en memoria

predictions = mod.ModelPredict(X_test)
post_pred = mod.post_processing(predictions)
mod.diceMetrics(y_test,post_pred,"full_model")
mod.print_examples(X_test,post_pred,{0,10,15})


