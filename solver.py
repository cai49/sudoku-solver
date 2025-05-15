# import the necessary packages
from modules.puzzle.puzzle import extract_digit, find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
from sudoku.sudoku import UnsolvableSudoku
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained digit classifier")
ap.add_argument("-i", "--image", required=True,
                help="path to input Sudoku puzzle image")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())

# load the digit classifier from disk
print("[INFO] loading digit classifier...")
model = load_model(args["model"])

# load the input image from disk and resize it
print("[INFO] processing image...")
alpha = 1.0
beta = 0.0
while True:
    image = cv2.imread(args["image"])
    try:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        image = imutils.resize(image, width=756)

        # find the puzzle in the image and then
        (puzzleImage, warped) = find_puzzle(image, debug=args["debug"] > 0)

        # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        # pos_warped = cv2.dilate(warped, dilate_kernel, iterations=1)
        # warped = warped + pos_warped

        # blur_kernel = kernal = np.array([[1/16,1/8,1/16], [1/8,1/4,1/8], [1/16,1/8,1/16]])
        # warped = cv2.filter2D(warped, -1, blur_kernel)
        warped = cv2.threshold(warped, 150, 255, cv2.THRESH_BINARY)[1]
        # warped = cv2.dilate(warped, dilate_kernel, iterations=1)
        # cv2.imshow("Warped", warped)
        # cv2.waitKey(0)

        # initialize our 9x9 Sudoku board
        board = np.zeros((9, 9), dtype="int")

        # a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
        # infer the location of each cell by dividing the warped image
        # into a 9x9 grid
        stepX = warped.shape[1] // 9
        stepY = warped.shape[0] // 9

        # initialize a list to store the (x, y)-coordinates of each cell
        # location
        cellLocs = []

        # loop over the grid locations
        for y in range(0, 9):
            # initialize the current list of cell locations
            row = []
            for x in range(0, 9):
                # compute the starting and ending (x, y)-coordinates of the
                # current cell
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY

                # add the (x, y)-coordinates to our cell locations list
                row.append((startX, startY, endX, endY))

                # crop the cell from the warped transform image and then
                # extract the digit from the cell
                cell = warped[startY:endY, startX:endX]
                digit = extract_digit(cell, debug=args["debug"] > 0)

                # verify that the digit is not empty
                if digit is not None:
                    # resize the cell to 28x28 pixels and then prepare the
                    # cell for classification
                    
                    kernal = np.array([[1/8,1/4,1/8], [1/4,1/2,1/4], [1/8,1/4,1/8]])
                    roi = cv2.filter2D(digit, -1, kernal)
                    roi = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)[1]
                    
                    roi = cv2.resize(roi, (28, 28))
                    cv2.imshow("Thresholded", roi)

                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # classify the digit and update the Sudoku board with the
                    # prediction
                    pred = model.predict(roi).argmax(axis=1)[0]
                    board[y, x] = pred + 1
                    print(pred + 1)

            # add the row to our cell locations
            cellLocs.append(row)

        # construct a Sudoku puzzle from the board
        print("[INFO] OCR'd Sudoku board:")
        puzzle = Sudoku(3, 3, board=board.tolist())
        puzzle.show()
        # solve the Sudoku puzzle
        print("[INFO] solving Sudoku puzzle...")

        solution = puzzle.solve(assert_solvable=True)
        solution.show_full()

        # loop over the cell locations and board
        for (cellRow, boardRow) in zip(cellLocs, solution.board):
            # loop over individual cell in the row
            for (box, digit) in zip(cellRow, boardRow):
                # unpack the cell coordinates
                startX, startY, endX, endY = box

                # compute the coordinates of where the digit will be drawn
                # on the output puzzle image
                textX = int((endX - startX) * 0.4)
                textY = int((endY - startY) * -0.3)
                textX += startX
                textY += endY

                # draw the result digit on the Sudoku puzzle image
                if digit:
                    cv2.putText(puzzleImage, str(digit), (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # show the output image
        cv2.imshow("Sudoku Result", puzzleImage)
        cv2.waitKey(0)
        break
    except UnsolvableSudoku:
        alpha = min(2.0, alpha + .075)
        beta = min(-0.5, beta - 0.01)
        # cv2.imshow("Show", image)
        # cv2.waitKey(0)
    except Exception as e:
        print(type(e))
        break
