import cv2
import numpy as np
import random


def gamma_correction(image, gamma):
    """
    Apply gamma correction to an image.
    
    Parameters:
        image (numpy.ndarray): Input image in BGR format.
        gamma (float): Gamma value for correction. Values > 1 will darken the image,
                       while values < 1 will brighten it.
                       
    Returns:
        numpy.ndarray: Gamma-corrected image.
    """
    # Build a lookup table mapping pixel values [0, 255] to their gamma-corrected values
    inv_gamma = 1.0 / gamma
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    
    # Apply the gamma correction using the lookup table
    corrected_image = cv2.LUT(image, lookup_table)
    
    return corrected_image



def draw_to_paper(cv2_color_img, cv2_paper_img):

    img = cv2_color_img.copy()
    height,width,z=img.shape

    paper_img = cv2.cvtColor(cv2_paper_img, cv2.COLOR_RGB2GRAY)
    paper_img = cv2.resize(paper_img, (width, height))

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y][x] = np.round(img[y][x] * (paper_img[y][x] / 255)) # change color by paper gray amplitude


    return gamma_correction(img, 1.4)






