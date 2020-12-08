import tensorflow
from tensorflow.keras.layers import Input, Subtract, Dense, Lambda, Conv2D, MaxPooling2D, Flatten, dot
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
import pickle
import os
import numpy as np
import sklearn


basefolder = r"C:\Users\karamvenkatsaigiridh\Desktop\Django\SingerAnalysis\Singers"
trainDir = r"C:\Users\karamvenkatsaigiridh\Desktop\Django\SingerAnalysis\trainData"

singerDirs = os.listdir(basefolder)
x = []
y = []
for singer in singerDirs:
    trainFile = trainDir + "/" + singer
    temp_x = pickle.load(open(trainFile + ".feature","rb"))
    temp_y = pickle.load(open(trainFile + ".label","rb"))

    x.append(temp_x)
    y.append(temp_y)


x = np.array(x)
y = np.array(y)

print(x.shape)

# max_x = np.amax(x.any())
# x = x/max_x

x_train, x_test, y_train, y_test = sklearn.train_test_split(x, y, test_size=0.15, random_state=0)
x_train, x_val, y_train, y_val = sklearn.train_test_split(x_train, y_train, test_size=0.25, random_state=0)

x_dummy = x_train[:5]
y_dummy = y_train[:5]

def get_siamese_model(input_shape):
    """
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                     kernel_initializer="random_uniform", kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer="random_uniform",
                     bias_initializer="zeros", kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer="random_uniform",
                     bias_initializer="zeros", kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer="random_uniform",
                     bias_initializer="zeros", kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer="random_uniform", bias_initializer="zeros"))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    similarity = dot([encoded_l,encoded_r], axes=-1, normalize=True)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=similarity)

    # return the model
    return siamese_net

optimizer = optimizers.Adam(lr = 0.00006)
model = get_siamese_model((10,10))
model.compile(loss="mean_squared_error",optimizer=optimizer)

model.fit(x_dummy,y_dummy,
          batch_size=128,
          epochs=25,
          verbose=2,
          validation_data=(x_val,y_val))
