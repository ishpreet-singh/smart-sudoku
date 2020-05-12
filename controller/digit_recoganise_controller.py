import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import backend as K

class DigitRecoganiseController:
    def __init__(self, pool_size, rate, number_neurons_output, epochs):
        self.pool_size = pool_size # (2, 2)
        self.rate = rate # 0.2
        self.number_neurons_output = number_neurons_output #10
        self.epochs = epochs

    def initialize_cnn(self):
        model = Sequential()
        ## Conv2D
        ## filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
        model.add(Conv2D(32, (5,5), input_shape = (28, 28, 1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = self.pool_size))
        model.add(Conv2D(16, (3,3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = self.pool_size))
        model.add(Dropout(self.rate))
        model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(self.number_neurons_output, activation = 'softmax'))
        return model

    def fit_model(self, model, X_train, y_train, X_test, y_test):
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = self.epochs, batch_size=200)
        return model

    def evaluate_model(self, model, x_test, y_test):
        return model.evaluate(x_test, y_test)
        
    def predict(self, model, x_test, y_test, image_index, img_rows, img_cols):
        plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
        pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
        print(pred.argmax())


if __name__ == "__main__":
    seed = 7
    np.random.seed(seed)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    number_neurons_output = y_test.shape[1]

    pool_size = (2, 2)
    rate = 0.2
    epochs = 10
    
    recognizer = DigitRecoganiseController(pool_size, rate, number_neurons_output, epochs)
    model = recognizer.initialize_cnn()
    model = recognizer.fit_model(model, X_train, y_train, X_test, y_test)
    print(recognizer.evaluate_model(model, X_test, y_test))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    