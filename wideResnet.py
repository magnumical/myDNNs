from keras.models import Model
from keras.layers import Dense, Dropout, Add, Input, BatchNormalization, Activation
from keras.layers import  Conv2D, AveragePooling2D, Flatten

def main_block(x, filters, n, strides, dropout):
	# Normal part
	x_res = Conv2D(filters, (3,3), strides=strides, padding="same")(x)
	x_res = BatchNormalization()(x_res)
	x_res = Activation('relu')(x_res)
	x_res = Conv2D(filters, (3,3), padding="same")(x_res)
	x = Conv2D(filters, (1,1), strides=strides)(x)
	x = Add()([x_res, x])

	for i in range(n-1):
		x_res = BatchNormalization()(x)
		x_res = Activation('relu')(x_res)
		x_res = Conv2D(filters, (3,3), padding="same")(x_res)
		if dropout: x_res = Dropout(dropout)(x)
		x_res = BatchNormalization()(x_res)
		x_res = Activation('relu')(x_res)
		x_res = Conv2D(filters, (3,3), padding="same")(x_res)
		x = Add()([x, x_res])

	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def build_wrn(input_dims, output_dim, n, k,filters=[16,32,64] ,act= "relu", dropout=None):

	assert (n-4)%6 == 0
	assert k%2 == 0
	n = (n-4)//6 
	inputs = Input(shape=(input_dims))

	x = Conv2D(16, (3,3), padding="same")(inputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = main_block(x, filters[0]*k, n, (1,1), dropout) # 0
	x = main_block(x, filters[1]*k, n, (2,2), dropout) # 1
	x = main_block(x, filters[2]*k, n, (2,2), dropout) # 2
			
	x = AveragePooling2D((8,8))(x)
	x = Flatten()(x)
	outputs = Dense(output_dim, activation="softmax")(x)

	model = Model(inputs=inputs, outputs=outputs)
	return model








