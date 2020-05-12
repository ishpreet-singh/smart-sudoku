# Smart Sudoku

A Smart Sudoku that provides the solution to a classical 9X9 Sudoku from its image using OpenCV and Deep Learning.

* Image Recoginition is done using OpenCV using techniques like Blurring, Adaptive Threhsolding and Inversion.

* A digit recoginition CNN model was trained on MNIST dataset. 
  
* The model recoginises the digit from the image. 
  
* Another CNN model was used to find the solution. 
  
* The second CNN model was trained on 1 million Sudoku Dataset which can be downloaded from [Kaggle](https://www.kaggle.com/bryanpark/sudoku)


## Stages

1. Stage 1 Cropped Image
   
![Cropped Image](https://github.com/ishpreet-singh/smart-sudoku/assets/stage1.png)

2. Stage 2 Image after Blurring, Thresholding, Inversion and Grid Detection

![Image after Preprocessing](https://github.com/ishpreet-singh/smart-sudoku/assets/stage2.png)

3. Stage 3 Image after Digit Reciginition

![Image after Digit Reciginition](https://github.com/ishpreet-singh/smart-sudoku/assets/stage3.png)

4. Stage 4 Final Solution

![Solution](https://github.com/ishpreet-singh/smart-sudoku/assets/stage4.png)


## Steps to run

1. Clone the project
    ```
    git clone https://github.com/ishpreet-singh/smart-sudoku
    ```

2. Move to project directory
    ```
    cd smart-sudoku
    ```

3. Create a Virtual Environment named `venv`. Read more about [Virtual Environment](https://docs.python.org/3/library/venv.html)

    ```
    virtualenv -p /path/to/python3 venv
    ```

4. Activate the virtual environment

    ```
    source venv/bin/activate
    ```

5. Run the Project

    ```
    python sudoku.py <IMAGE_PATH>
    ```
    
    >***Note***: You need to provide the path of your sudoku image.
    > *Eg*: `python sudoku.py dataset/sudoku2.jpg`  

## Project Structure

```
- smart-sudoku/
   - LICENSE
   - README.md
   - assets/
     - stage1.png
     - stage2.png
     - stage3.png
     - stage4.png
   - controller/
     - __init__.py
     - cnn.model
     - cnn.py
     - digit_recoganise_controller.py
     - gpus.py
     - image_controller.py
     - model.h5
     - model.json
     - sudoku_controller.py
     - sudoku_solver.py
   - dataset/
     - mnist.npz
     - sudoku.csv
     - sudoku1.jpg
     - sudoku2.jpg
     - sudoku3.jpg
   - requirements.txt
   - sudoku.py
```

## File Description

