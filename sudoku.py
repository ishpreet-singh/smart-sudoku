import sys
from controller.image_controller import ImageController
from controller.sudoku_controller import SudokuController
from keras.models import model_from_json

if __name__ == "__main__":
    print("Insdie Sudoku.py")

    ic = ImageController() 

    #Load the saved model
    json_file = open('controller/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("controller/model.h5")
    print("Loaded saved model from disk.")

    path = 'dataset/Sample1.jpg'
    image = ic.controller(path)

    grid = ic.extract_number(image, loaded_model)
    sc = SudokuController(grid)
    ic.display_sudoku(grid.tolist())

    solution = sc.sudoku_solver(grid)
    print('Solution:')
    ic.display_sudoku(solution.tolist())