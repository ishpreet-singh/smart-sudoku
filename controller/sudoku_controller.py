
class SudokuController:

    def __init__(self, grid):
        self.grid = grid
        self.current_row = 0
        self.current_col = 0


    def show_grid(self):
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == 0:
                    print(".", end = " ")
                else:
                    print(self.grid[row][col], end=" ")
            print("\n")


    def is_cell_empty(self):
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == 0:
                    self.current_row = row
                    self.current_col = col
                    return True
        return False


    def is_row_valid(self, row, num):
        for c in range(9):
            if self.grid[row][c] == num:
                return False
        return True


    def is_col_valid(self, col, num):
        for r in range(9):
            if self.grid[r][col] == num:
                return False
        return True


    def is_sub_grid_valid(self, row, col, num):
        for r in range(3):
            for c in range(3):
                if self.grid[r + row][c + col] == num:
                    return False
        return True


    def is_cell_valid(self, row, col, num):
        return self.is_row_valid(row, num) and self.is_col_valid(col, num) and self.is_sub_grid_valid(row - row % 3, col - col % 3, num)


    def mark_cell(self, row, col, num):
        self.grid[row][col] = num


    def unmark_cell(self, row, col):
        self.grid[row][col] = 0


    def solve(self,grid):

        if(not self.is_cell_empty()):
            return True

        row = self.current_row
        col = self.current_col

        for num in range(1, 10):
            if self.is_cell_valid(row, col, num):
                self.mark_cell(row, col, num)
                if self.solve(grid):
                    return True
                self.unmark_cell(row, col)
        return False

    def sudoku_solver(self, grid):
        if(self.solve(grid)):
            print('---') # print_grid(grid) -> to print the sudoku elements
        else:
            print ("No solution exists")
        grid = grid.astype(int)
        return grid

if __name__ == "__main__":
    print("Insdie Sudoku Controller.py")
    grid = [[3, 0, 6, 5, 0, 8, 4, 0, 0],
            [5, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 8, 7, 0, 0, 0, 0, 3, 1],
            [0, 0, 3, 0, 1, 0, 0, 8, 0],
            [9, 0, 0, 8, 6, 3, 0, 0, 5],
            [0, 5, 0, 0, 9, 0, 6, 0, 0],
            [1, 3, 0, 0, 0, 0, 2, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 7, 4],
            [0, 0, 5, 2, 0, 6, 3, 0, 0]]
    sc = SudokuController(grid)
    print("Original Sudoku")
    sc.show_grid()
    if sc.solve(grid):
        print("Solution: \n")
        sc.show_grid()
    else:
        print("Solution doesn't exist!")
