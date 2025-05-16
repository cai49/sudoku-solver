from modules.puzzle.puzzle import extract_digit, find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
from sudoku.sudoku import UnsolvableSudoku
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained digit classifier")
ap.add_argument("-i", "--image", required=True,
                help="path to input Sudoku puzzle image")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())

print("[INFO] Now loading keras model")
model = load_model(args["model"])

print("[INFO] Running iterative loop")
alpha = 1.0
beta = 0.0
while True:
    image = cv2.imread(args["image"])
    try:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        image = imutils.resize(image, width=756)

        (puzzleImage, warped) = find_puzzle(image, debug=args["debug"] > 0)

        warped = cv2.threshold(warped, 150, 255, cv2.THRESH_BINARY)[1]

        board = np.zeros((9, 9), dtype="int")

        stepX = warped.shape[1] // 9
        stepY = warped.shape[0] // 9

        cellLocs = []

        for y in range(0, 9):
            row = []
            for x in range(0, 9):
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY

                row.append((startX, startY, endX, endY))

                cell = warped[startY:endY, startX:endX]
                digit = extract_digit(cell, debug=args["debug"] > 0)

                if digit is not None:
                    # Gaussian blur filtering stage and apply a threshold
                    kernal = np.array([[1/8,1/4,1/8], [1/4,1/2,1/4], [1/8,1/4,1/8]])
                    roi = cv2.filter2D(digit, -1, kernal)
                    roi = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)[1]

                    roi = cv2.resize(roi, (28, 28))

                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    pred = model.predict(roi).argmax(axis=1)[0]
                    board[y, x] = pred + 1

            cellLocs.append(row)

        puzzle = Sudoku(3, 3, board=board.tolist())
        puzzle.show()

        print("[INFO] Solving...")
        solution = puzzle.solve(assert_solvable=True)
        solution.show_full()

        for (cellRow, boardRow) in zip(cellLocs, solution.board):
            for (box, digit) in zip(cellRow, boardRow):
                startX, startY, endX, endY = box
                textX = 0
                textY = 0
                textX += startX
                textY += endY

                if digit:
                    cv2.rectangle(puzzleImage, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(puzzleImage, str(digit), (textX, textY),
                                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Result", puzzleImage)
        cv2.waitKey(0)
        break
    except UnsolvableSudoku:
        alpha = min(2.0, alpha + .075)
        beta = min(-0.5, beta - 0.01)
    except Exception as e:
        print(type(e))
        break
