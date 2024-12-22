import cv2
import numpy as np


def rolling_edge_detection(cv2_img, iteration):

    blurred = cv2.GaussianBlur(cv2_img, (3, 3), 0) #去除雜質
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)

    edge_output = cv2.Canny(gray, 60, 110, apertureSize=3)
    canny = cv2.cvtColor(edge_output, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f"Canny_{0}.png", canny)

    for i in range(0, iteration -1):
        edge_output = cv2.Canny(edge_output, 60, 110, apertureSize=3)
        new_canny = cv2.cvtColor(edge_output, cv2.COLOR_BGR2RGB)
        new_canny = cv2.add(new_canny, canny)  

        canny = new_canny
        # cv2.imwrite(f"Canny_{i+1}.png", canny)

    return cv2.add(cv2_img, canny)