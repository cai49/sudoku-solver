from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_puzzle(image, debug=False):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    puzzle_contour = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            puzzle_contour = approx
            break

    if puzzle_contour is None:
        raise Exception("Could not find Sudoku puzzle outline. ")

    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzle_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", output)
        cv2.waitKey(0)

    puzzle = four_point_transform(image, puzzle_contour.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_contour.reshape(4, 2))

    if debug:
        cv2.imshow("Transform", puzzle)
        cv2.waitKey(0)

    return puzzle, warped

def extract_digit(cell, debug=False):
    thresh = cv2.threshold(cell, 10, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, [255], -1)

    (h, w) = thresh.shape
    percent_filled = cv2.countNonZero(mask) / float(w * h)

    if percent_filled < 0.03:
        return None

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)

    return digit