from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D 
from keras.layers import AveragePooling2D


def lenet5_builder(input_shape,n_cls):
    
    model = Sequential()
    
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1),
                            activation='tanh', input_shape=input_shape,
                            padding='same'))
    
    
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    
    
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    
    
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    
    model.add(Flatten())
    
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(n_cls, activation='softmax'))
    

    return model







