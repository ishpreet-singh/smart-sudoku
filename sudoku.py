import sys
import cv2
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
from keras.models import model_from_json
import keras.backend.tensorflow_backend as tfback
from controller.gpus import get_available_gpus
from controller.sudoku_solver import SudokuSolver
from controller.image_controller import ImageController
from controller.sudoku_controller import SudokuController

if __name__ == "__main__":
    tfback._get_available_gpus = get_available_gpus

    try:
        start_time = time()

        # Load the saved model
        json_file = open('controller/model.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("controller/model.h5")
        print("Load saved model from disk.")

        # Image Processing
        ic = ImageController() 
        img_path = sys.argv[1]
        image = ic.controller(img_path)

        # Extract Numbers from Grid
        grid = ic.extract_number(image, model)
        sc = SudokuController(grid)
        ic.display_sudoku(grid.tolist())

        # Solution using CNN
        ss = SudokuSolver()
        solution = ss.solve_sudoku(grid)

        # Solution using Backtracting
        # solution = sc.sudoku_solver(grid)
        
        print('\nSolution:')
        ic.display_sudoku(solution.tolist())

        end_time = time()
        print(f"\nTotal Time: {round(end_time - start_time, 3)}s")

    except:
        # Some Error Happened
        fmt = 'usage: {} image_path'
        print(fmt.format(__file__.split('/')[-1]))
        print('[ERROR]: Image not found')

