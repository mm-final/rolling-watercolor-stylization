import cv2
import numpy as np
import random
import image_dithering


def rolling_edge_detection(cv2_img, iteration):

    blurred = cv2.GaussianBlur(cv2_img, (3, 3), 0) #去除雜質
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)

    edge_output = cv2.Canny(gray, 60, 110, apertureSize=3)
    canny = cv2.cvtColor(edge_output, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f"Canny_{0}.png", canny)

    for i in range(0, iteration -1):
        edge_output = cv2.Canny(edge_output, 60, 110, apertureSize=3)
        new_canny = cv2.cvtColor(edge_output, cv2.COLOR_BGR2RGB)
        new_canny = ditering(new_canny, i * 0.2)
        
        new_canny = cv2.add(new_canny, canny)  

        canny = new_canny
        # cv2.imwrite(f"Canny_{i+1}.png", canny)

    return canny


# def edge_seperation(edge_img):

#     img = edge_img.copy()

#     for y in range(img.shape[0]):
#         is_filling_x = False
#         is_start_fill_x = False

#         for x in range(img.shape[1]):
#             if not is_start_fill_x:
#                 if img[y][x][0] == 255: # edge
#                     is_filling_x = True
#                     is_start_fill_x = True
#             else:
#                 is_start_fill_x = True
#                 now_color_x = img[y][x][0]
#             else:
#                 if is_filling_x:
#                     img[y][x] = [255, 255, 255]

#     return img


def ditering(cv2_img, diter_chance):

    # new_img = cv2_img.copy()
    
    # for y in range(new_img.shape[0]):
    #     for x in range(new_img.shape[1]):
    #         # if (new_img[y][x] == 255).all():
    #         if random.random() < diter_chance:
    #             new_img[y][x] = np.clip(np.round(new_img[y][x] * 1.75), 0, 255) #  randomly ditering

    


    # img = cv2_img.copy()

    # width,height,z=img.shape
    # w1=7/16.0
    # #print w1
    # w2=3/16.0
    # w3=5/16.0
    # w4=1/16.0

    # gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blue=img[:,:,0]  #taking the blue channel
    # blue=image_dithering.stucki(blue)   #sending it to stucki algorithm
    # blue=image_dithering.stucki(blue)   #histogram equalising the result  same applies for remaining channels below

    # green=img[:,:,1]
    # green=image_dithering.stucki(green)
    # green=image_dithering.stucki(green)

    # red=img[:,:,2]
    # red=image_dithering.stucki(red)
    # red=image_dithering.stucki(red)

    # img = cv2.merge((blue, green, red))  #merging the 3 color channels





    img = cv2_img.copy()
    height,width,z=img.shape

    paper_img = cv2.imread("oil_paper.png")
    paper_img = cv2.cvtColor(paper_img, cv2.COLOR_RGB2GRAY)
    paper_img = cv2.resize(paper_img, (width, height))

    # # Generate fixed pattern noise
    # # Use Gaussian distribution; mean=0, std=10
    # fpn = np.random.normal(0, 10, (height, width)).astype(np.float32)
    # # Normalize to range 0-255
    # fpn_normalized = cv2.normalize(fpn, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # fpn_normalized = fpn_normalized.astype(np.uint8)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # img[y][x] = np.clip(np.round(img[y][x]* 1.75), 0, 255) #  randomly ditering

            img[y][x] = np.round(img[y][x] * (paper_img[y][x] / 255))  #  randomly ditering

            ##########################



    return gamma_correction(img, 1.4)
    # return img






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