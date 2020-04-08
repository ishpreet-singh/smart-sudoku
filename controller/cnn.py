import keras
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape


class CNN:

    def __init__(self):

        self.batch_size = 64
        self.epoch = 2


    def train(self, x_train, y_train):

        self.model = keras.models.Sequential()

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(81*9))
        self.model.add(Reshape((-1, 9)))
        self.model.add(Activation('softmax'))

        adam = keras.optimizers.adam(lr=.001)
        self.model.compile( loss='sparse_categorical_crossentropy', optimizer=adam)

        print(self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epoch))

        self.model.save('cnn.model')

if __name__ == "__main__":
    pass
