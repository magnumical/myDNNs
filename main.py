
'''
This is the database of various models I used in my codes!
Ithought hat it would be great if I gather all models here
to fascilate the coding progress in my future apps!

- Reza Amini
imreza.ir
github.com/magnumical
twitter.com/reza__amini

'''
from keras.datasets import cifar10
from keras.utils import np_utils
import keras

from LeNet5 import *
from wideResnet import *
from Res_net import *

from vgg_16 import *
from vgg_19 import *

#%% 
#       CIFAR10 dataset is loaded to test the models!

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')/ 255.0
X_test = X_test.astype('float32')/ 255.0


Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#################################################
#################################################
# these two following variales are important!!!!
input_shape = (32, 32, 3)
n_cls= 10
print("CIFAR is just LOADED!")



#%% Build Lenet-5

def model_selector(input_shape, n_cls):

    n_model = input("Select model (Enter number) \n 1. Lenet-5 \n 2. ResNet \n 3. WRN\n 4. VGG-16 \n 5. VGG-19 \n ")
    
    if int(n_model)==1:
        model= lenet5_builder(input_shape,n_cls)
        
    if int(n_model)==2:
        whichRN = int(input("Select model (Enter number) \n 1. RN18 \n 2. RN50 \n 3. RN101 \n 4. RN150 \n "))
        
        if whichRN==1: 
            model= resnet_build18(input_shape,n_cls)
        if whichRN==2: 
            model= resnet_build50(input_shape,n_cls)        
        if whichRN==3: 
            model= resnet_build101(input_shape,n_cls)
        if whichRN==4: 
            model= resnet_build150(input_shape,n_cls)
            
    if int(n_model)==3:
        d = int(input("Enter desired depth you want (Enter number, e.g 28) : \n "))
        k = int(input("Enter desired scale you want (Enter number e.g. 8) : \n "))
        print("I also set filter sizes as 16,32,64 \n")

        model= build_wrn(input_shape, n_cls, d, k,filters=[16,32,64])

        
    if int(n_model)==4:
        model= vgg16_builder(input_shape,n_cls)
        
    if int(n_model)==5:
        model= VGG_19(input_shape,n_cls)
        
  
    return model



model = model_selector(input_shape, n_cls)



#%%
model.compile(loss= keras.losses.categorical_crossentropy,
              optimizer= keras.optimizers.SGD(),
              metrics=['accuracy'])
model.fit(x=X_train, y=Y_train, batch_size=1024,
          epochs=1,validation_split=0.1)    




















