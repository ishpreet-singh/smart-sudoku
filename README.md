# smart-sudoku
AI based Sudoku Solver


1. Digit recognaizer
2. Sudoko Image Extraction
3. Suduko Solution using Backtracking
4. Suduko Solution using CNN, KNN
5. Python UI using TkInter


Structure

+ dataset -> Ish
  + mnist Dataset
  + images minimum 4-5 Advisable 10-20
  + Sample Sudoku
  + Sample Soduko Solution
+ gitignore -> Ish
+ venv 
+ requirements.txt -> Ish 
+ sudoku.py ~ main.py -> Ish
+ Models/ -> Sam
  + Data Set Saved Model
+ View/ -> Ish
  + UI for Sudoku ~ TK Inter
  + View.py ~ Tk Inter
+ Controllers/
  + Digit recoganizer -> Sam
    + Review Ish or Harry's Branch 
    + Model Should Train only Once
  + Image Controller -> Harry
    + Preprocessing
    + Countour Handling
    + Cell Extraction
  + Sukoku Manager or Sudoku Controller -> Ish
    + Manages Image Controller and Digit recoganizer
    + Uses input fetched from View.py

