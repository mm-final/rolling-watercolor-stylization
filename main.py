import cv2
import numpy as np
import matplotlib.pyplot as plt
import stroke
from permutohedral import PermutohedralLattice


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w,
                                        self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w,
                                             self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        # setup a look-up table for spatial kernel
        LUT_s = np.exp(-0.5*(np.arange(self.pad_w+1)**2)/self.sigma_s**2)
        # setup a look-up table for range kernel
        LUT_r = np.exp(-0.5*(np.arange(256)/255)**2/self.sigma_r**2)
        # compute the weight of range kernel by rolling the whole image
        wgt_sum, result = np.zeros(
            padded_img.shape), np.zeros(padded_img.shape)
        for x in range(-self.pad_w, self.pad_w+1):
            for y in range(-self.pad_w, self.pad_w+1):
                # method 1 (easier but slower)
                dT = LUT_r[np.abs(
                    np.roll(padded_guidance, [y, x], axis=[0, 1])-padded_guidance)]
                r_w = dT if dT.ndim == 2 else np.prod(
                    dT, axis=2)  # range kernel weight
                s_w = LUT_s[np.abs(x)]*LUT_s[np.abs(y)]  # spatial kernel
                t_w = s_w*r_w
                padded_img_roll = np.roll(padded_img, [y, x], axis=[0, 1])
                for channel in range(padded_img.ndim):
                    result[:, :, channel] += padded_img_roll[:, :, channel]*t_w
                    wgt_sum[:, :, channel] += t_w
        output = (result/wgt_sum)[self.pad_w:-
                                  self.pad_w, self.pad_w:-self.pad_w, :]

        return np.clip(output, 0, 255).astype(np.uint8)

    def permutohedralfilter(self, img, ref):
        invSpatialStdev = float(1.0/self.sigma_s)
        invColorStdev = float(1.0/self.sigma_r)

        # Construct the position vectors out of x, y, r, g, and b.
        height = img.shape[0]
        width = img.shape[1]

        eCh = ref.shape[2]
        iCh = img.shape[2]
        positions = np.zeros((height, width, 2+eCh), dtype=np.float32)

        img = np.array(img) / 255.

        # From Mat to Image
        for y in range(height):
            for x in range(width):
                positions[y, x, 0] = invSpatialStdev * x
                positions[y, x, 1] = invSpatialStdev * y

                for c in range(eCh):
                    positions[y, x, 2+c] = invColorStdev * img[y, x, c]

        out = PermutohedralLattice.filter(img, positions)

        # Save the result
        imgOut = np.zeros(img.shape, dtype=img.dtype)
        for y in range(height):
            for x in range(width):
                for c in range(iCh):
                    imgOut[y, x, c] = out[y, x, c]

        return imgOut


image_dir = "images"
image_name = "image.png"

origin_img = cv2.imread(f"{image_dir}/{image_name}")

b, g, r = cv2.split(origin_img)
origin_img = cv2.merge([r, g, b])

temp = origin_img.copy()

iteration_time = 4
sigma_s = 3
sigma_r = 25.5
filter = Joint_bilateral_filter(sigma_s, sigma_r)


for i in range(iteration_time):
    if i == 0:
        temp = cv2.GaussianBlur(origin_img, (0, 0), sigma_s, sigma_s)
    else:
        temp = filter.permutohedralfilter(origin_img, temp)


# permutohedralfilter output normalize `float64`, cast back to `uint8`
temp = (temp * 255).astype(np.uint8)

# add stroke by rolling edge detecion
temp = stroke.rolling_edge_detection(temp, 4)


plt.imshow(temp)
plt.show()
