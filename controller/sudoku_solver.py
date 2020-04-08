import numpy as np
import pandas as pd
from controller.cnn import CNN
import keras
import os 
from sklearn.model_selection import train_test_split
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape


class SudokuSolver:

    def __init__(self, train = False):
        if not train:
            self.load_model()
        pass


    def norm(self, a):
        return (a/9)-.5


    def denorm(self, a):
        return (a+.5)*9
        

    def get_data_set(self):
        '''
            Loading 1M Sudoku's as Dataset, This will take some time Hang On
        '''
        dataset_path = "../dataset/sudoku.csv"
        data = pd.read_csv(dataset_path)

        feat_raw = data['quizzes']
        label_raw = data['solutions']

        feat = []
        label = []

        for i in feat_raw:

            x = np.array([int(j) for j in i]).reshape((9, 9, 1))
            feat.append(x)

        feat = np.array(feat)
        feat = feat/9
        feat -= .5

        for i in label_raw:

            x = np.array([int(j) for j in i]).reshape((81, 1)) - 1
            label.append(x)

        label = np.array(label)

        del(feat_raw)
        del(label_raw)

        x_train, x_test, y_train, y_test = train_test_split(
            feat, label, test_size=0.2, random_state=42)

        return x_train, x_test, y_train, y_test


    def train_model(self):

        x_train, x_test, y_train, y_test = self.get_data_set()
        self.model = CNN()
        self.model.train(x_train, y_train)
        

    def load_model(self):
        model_path = os.path.join(os.getcwd() , "controller/cnn.model")
        self.model = keras.models.load_model(model_path)


    def solve(self, sample):
        '''
            This function solve's the sudoku by filling blank positions one by one.
        '''
        feat = sample

        while(1):

            out = self.model.predict(feat.reshape((1, 9, 9, 1)))
            out = out.squeeze()

            pred = np.argmax(out, axis=1).reshape((9, 9))+1
            prob = np.around(np.max(out, axis=1).reshape((9, 9)), 2)

            feat = self.denorm(feat).reshape((9, 9))
            mask = (feat == 0)

            if(mask.sum() == 0):
                break

            prob_new = prob*mask

            ind = np.argmax(prob_new)
            x, y = (ind//9), (ind % 9)

            val = pred[x][y]
            feat[x][y] = val
            feat = self.norm(feat)

        return pred


    def test_accuracy(self, feats, labels):

        correct = 0

        for i, feat in enumerate(feats):

            pred = self.solve(feat)

            true = labels[i].reshape((9, 9))+1

            if(abs(true - pred).sum() == 0):
                correct += 1

        print(correct/feats.shape[0])


    def solve_sudoku(self, game):
        game = game.reshape((9, 9, 1))
        game = self.norm(game)
        game = self.solve(game)
        return game


if __name__ == "__main__":

    ss = SudokuSolver()
    game = [ [0, 8, 0, 0, 3, 2, 0, 0, 1],
            [7, 0, 3, 0, 8, 0, 0, 0, 2],
            [5, 0, 0, 0, 0, 7, 0, 3, 0],
            [0, 5, 0, 0, 0, 1, 9, 7, 0],
            [6, 0, 0, 7, 0, 9, 0, 0, 8],
            [0, 4, 7, 2, 0, 0, 0, 5, 0],
            [0, 2, 0, 6, 0, 0, 0, 0, 9],
            [8, 0, 0, 0, 9, 0, 3, 0, 5],
            [3, 0, 0, 8, 2, 0, 0, 1, 0]]

    game = np.array([int(j) for j in game]).reshape((9, 9, 1))

    # ss.train()
    ss.load_model()
    game = ss.solve_sudoku(game)

    print('solved puzzle:\n')
    print(game)
