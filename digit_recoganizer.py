
import os 
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

class DigitRecoganizer():

    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/mnist.npz")

    def __init__(self, gray_scale = True):
        
        if gray_scale == True:
            self.width = 28
            self.height = 28
            self.color = 1
        else:
            self.width = 128
            self.height = 128
            self.color = 3

        pass


    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data(self.dataset_path)


    def load_image_with_index(index = 7777):
        plt.imshow(x_train[image_index], cmap='Greys')
        plt.show()


    def normalize(self):
        # Reshaping the array to 4-dims so that it can work with the Keras API
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.width, self.height, self.color)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.width, self.height, self.color)
        self.input_shape = (self.width, self.height, self.color)

        # Making sure that the values are float so that we can get decimal points after division
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')

        # Normalizing the RGB codes by dividing it to the max RGB value.
        self.x_train /= 255
        self.x_test /= 255
        

    def cnn_model(self):
        # Creating a Sequential Model and adding the layers
        self.model = Sequential()
        self.model.add(Conv2D(28, kernel_size=(3,3), input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        self.model.add(Dense(128, activation=tf.nn.relu))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10,activation=tf.nn.softmax))


    def train(self, epoch = 1):
        self.epoch = epoch
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x=self.x_train,y=self.y_train, epochs=self.epoch)
        self.model.evaluate(self.x_test, self.y_test)


    def predict(self, image):
        plt.imshow(image.reshape(self.width, self.height),cmap='Greys')
        plt.show()
        pred = self.model.predict(image.reshape(1, self.width, self.height, self.color))
        print("Predict: ", pred)
        return pred.argmax()
        # print(pred.argmax())
    

if __name__ == "__main__":
    dg = DigitRecoganizer()
    dg.load_data()
    dg.normalize()
    dg.cnn_model()
    dg.train()
    image = dg.x_test[4444].reshape(28, 28)
    dg.predict(image)