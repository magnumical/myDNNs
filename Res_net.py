import keras
try:
   from keras_resnet import models
except ImportError:
   print('Error, Module keras_resnet is required')


def resnet_build18(input_shape,n_cls):

    shape, classes = input_shape, n_cls
    x = keras.layers.Input(shape)
    model = models.ResNet18(x, classes=classes)


    return model

#%% Resnet50
def resnet_build50(input_shape,n_cls):

    shape, classes = input_shape, n_cls
    x = keras.layers.Input(shape)
    model = models.ResNet50(x, classes=classes)
    return model

#%%
def resnet_build101(input_shape,n_cls):

    shape, classes = input_shape, n_cls
    x = keras.layers.Input(shape)
    model = models.ResNet101(x, classes=classes)
    return model

#%%
def resnet_build150(input_shape,n_cls):

    shape, classes = input_shape, n_cls
    x = keras.layers.Input(shape)
    model = models.ResNet152(x, classes=classes)
    return model







