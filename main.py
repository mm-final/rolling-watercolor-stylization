import cv2
import numpy as np
import matplotlib.pyplot as plt
import stroke
from permutohedral import PermutohedralLattice


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s

    def permutohedralfilter(self, img, ref):

        invSpatialStdev = float(1.0/self.sigma_s)
        invColorStdev = float(1.0/self.sigma_r)

        # Construct the position vectors out of x, y, r, g, and b.
        height = img.shape[0]
        width = img.shape[1]

        eCh = ref.shape[2]
        positions = np.zeros((height, width, 2+eCh), dtype=np.float32)

        # From Mat to Image
        for y in range(height):
            for x in range(width):
                positions[y, x, 0] = invSpatialStdev * x
                positions[y, x, 1] = invSpatialStdev * y
                positions[y, x, 2:-1] = invColorStdev * ref[y, x, :-1]

        out = PermutohedralLattice.filter(img, positions)

        return out * 255


image_dir = "images"
image_name = "scale_aware_g4.png"

origin_img = cv2.imread(f"{image_dir}/{image_name}")
origin_shape = origin_img.shape
origin_img = cv2.resize(origin_img, dsize=(
    512, 512), interpolation=cv2.INTER_LINEAR)

b, g, r = cv2.split(origin_img)
origin_img = cv2.merge([r, g, b])

iteration_time = 0
sigma_s = 3
sigma_r = 25.5
filter = Joint_bilateral_filter(sigma_s, sigma_r)

filter_buffer = []

for i in range(iteration_time):
    if i == 0:
        temp = cv2.GaussianBlur(origin_img, (0, 0), sigma_s, sigma_s)
        filter_buffer.append(temp.copy())
        origin_img = origin_img / 255.
        origin_img = origin_img.astype(np.float32)
    else:
        temp = filter.permutohedralfilter(origin_img, temp)

        if filter_buffer != []:
            for j in range(len(filter_buffer)):
                print(i, "==", j, ":", (temp == filter_buffer[j]).all())
        filter_buffer.append(temp.copy())

# fig, ax = plt.subplots(2, 2)
# for i in range(2):
#     for j in range(2):
#         print(filter_buffer[2*i+j].max(), filter_buffer[2*i+j].min())
#         ax[i, j].imshow(filter_buffer[2*i+j] / 255)

# permutohedralfilter output normalize `float64`, cast back to `uint8`
# temp = (filter_buffer[3]).astype(np.uint8)

# add stroke by rolling edge detecion
# temp = stroke.rolling_edge_detection(temp, 4)


BG_COLOR = 209
BG_SIGMA = 5
MONOCHROME = 1


def blank_image(width=1024, height=1024, background=BG_COLOR):
    """
    It creates a blank image of the given background color
    """
    img = np.full((height, width, MONOCHROME), background, np.uint8)
    return img


def add_noise(img, sigma=BG_SIGMA):
    """
    Adds noise to the existing image
    """
    width, height, ch = img.shape
    n = noise(width, height, sigma=sigma)
    img = img + n
    return img.clip(0, 255)


def noise(width, height, ratio=1, sigma=BG_SIGMA):
    """
    The function generates an image, filled with gaussian nose. If ratio parameter is specified,
    noise will be generated for a lesser image and then it will be upscaled to the original size.
    In that case noise will generate larger square patterns. To avoid multiple lines, the upscale
    uses interpolation.

    :param ratio: the size of generated noise "pixels"
    :param sigma: defines bounds of noise fluctuations
    """
    mean = 0
    assert width % ratio == 0, "Can't scale image with of size {} and ratio {}".format(
        width, ratio)
    assert height % ratio == 0, "Can't scale image with of size {} and ratio {}".format(
        height, ratio)

    h = int(height / ratio)
    w = int(width / ratio)

    result = np.random.normal(mean, sigma, (w, h, MONOCHROME))
    if ratio > 1:
        result = cv2.resize(result, dsize=(width, height),
                            interpolation=cv2.INTER_LINEAR)
    return result.reshape((width, height, MONOCHROME))


def texture(image, sigma=BG_SIGMA, turbulence=2):
    """
    Consequently applies noise patterns to the original image from big to small.

    sigma: defines bounds of noise fluctuations
    turbulence: defines how quickly big patterns will be replaced with the small ones. The lower
    value - the more iterations will be performed during texture generation.
    """
    result = image.astype(float)
    cols, rows, ch = image.shape
    ratio = cols
    while not ratio == 1:
        result += noise(cols, rows, ratio, sigma=sigma)
        ratio = (ratio // turbulence) or 1
    cut = np.clip(result, 0, 255)
    return cut.astype(np.uint8)


img = origin_img

img = texture(img, sigma=4, turbulence=2)

img = cv2.resize(img, dsize=(
    origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_LINEAR)

plt.imshow(img)
plt.show()
