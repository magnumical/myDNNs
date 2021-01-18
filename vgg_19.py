from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D 



def VGG_19(input_shape,n_cls):
    model = Sequential()
    # block 1
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding='same',
                     activation='relu',
                      input_shape=input_shape,
                     name='block1_conv1'))
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding='same',
                     activation='relu',
                     name='block1_conv2'))
    model.add(MaxPooling2D(pool_size=(2,2),
                           strides=(2,2),
                           name='block1_pool'))


    # block 2
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block2_conv1'))
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block2_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2,2),
                           name='block2_pool'))


    # block 3
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block3_conv1'))
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block3_conv2'))
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block3_conv3'))
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block3_conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2,2),
                           name='block3_pool'))


    # bvlock 4
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block4_conv1'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block4_conv2'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block4_conv3'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block4_conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2,2),
                           name='block4_pool'))


    # block 5
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block5_conv1'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block5_conv2'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block5_conv3'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block5_conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2,2),
                           name='block5_pool'))
    model.add(Flatten())
    model.add(Dense(units=4096,activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096,activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(units=n_cls,activation='softmax', name='predictions'))


    return model









