
class SudokuController:

    def __init__(self, grid):
        self.row_col = [0, 0]
        self.grid = grid

    def show_grid(self, grid):
        for row in range(9):
            for col in range(9):
                print(grid[row][col], end=" ")
            print("\n")

    def is_cell_empty(self, grid):
        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0:
                    self.row_col[0] = row
                    self.row_col[1] = col
                    return True
        return False

    def is_row_valid(self, grid, row, num):
        for c in range(9):
            if grid[row][c] == num:
                return True
        return False

    def is_col_valid(self, grid, col, num):
        for r in range(9):
            if grid[r][col] == num:
                return True
        return False

    def is_sub_grid_valid(self, grid, row, col, num):
        for r in range(3):
            for c in range(3):
                if grid[r + row][c + col] == num:
                    return True

        return False

    def is_cell_valid(self, grid, row, col, num):
        return not self.is_row_valid(grid, row, num) and not self.is_col_valid(grid, col, num) and not self.is_sub_grid_valid(grid, row - row%3, col - col%3, num)

    def mark_cell(self, grid, row, col, num):
        grid[row][col] = num

    def unmark_cell(self, grid, row, col):
        grid[row][col] = 0

    def solve(self, grid):

        if(not self.is_cell_empty(grid)):
            return True

        row = self.row_col[0]
        col = self.row_col[1]
        for num in range(1, 10):
            if self.is_cell_valid(grid, row, col, num):
                grid[row][col] = num
                if self.solve(grid):
                    return True
                grid[row][col] = 0
        return False


if __name__ == "__main__":
    print("Insdie Sudoku Controller.py")
    grid = [[0, 9, 0, 0, 0, 0, 8, 5, 3],
            [0, 0, 0, 8, 0, 0, 0, 0, 4],
            [0, 0, 8, 2, 0, 3, 0, 6, 9],
            [5, 7, 4, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 9, 0, 0, 6, 3, 7],
            [9, 4, 0, 1, 0, 8, 5, 0, 0],
            [7, 0, 0, 0, 0, 6, 0, 0, 0],
            [6, 8, 2, 0, 0, 0, 0, 9, 0]]
    sc = SudokuController(grid)
    print("Original Sudoku")
    sc.show_grid(grid)
    if sc.solve(grid):
        print("Sudoku is: ")
        sc.show_grid(grid)
    else:
        print("Solution doesn't exist!")
        print("Current Grid: \n", sc.show_grid(grid))
