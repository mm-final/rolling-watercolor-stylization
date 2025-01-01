import cv2
import numpy as np
import matplotlib.pyplot as plt
import paper
import dearpygui.dearpygui as dpg
from array import array
import noise_texture

dpg.create_context()
dpg.create_viewport(title='Custom Title', width=1024, height=512)


image_dir = "images"

image_name = "scale_aware_b4.png"
paper_name = "paper.jpg"

origin_img = cv2.imread(f"{image_dir}/{image_name}")

b, g, r = cv2.split(origin_img)
origin_img = cv2.merge([r, g, b])

iteration_time = 4
# b4 s 1 r 0-5
sigma_s = 10
sigma_r = 0.1
filter_buffer = []

origin_img = origin_img / 255.
origin_img = origin_img.astype(np.float32)


image_width, image_height, rgb_channel, rgba_channel = origin_img.shape[
    1], origin_img.shape[0], 3, 4
image_pixels = image_height * image_width
raw_data_size = image_width * image_height * rgba_channel

texture_format = dpg.mvFormat_Float_rgba

raw_data = array('f', [1] * raw_data_size)
origin_image = array('f', [1] * raw_data_size)

present_img = origin_img


with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        width=image_width, height=image_height, default_value=raw_data,
        format=texture_format, tag="image"
    )


def update_image(new_image):
    global present_img
    present_img = new_image


width, height = 512, 1024
posx, posy = 0, 0
color_change_debug = 0


def set_sigma_s(sender):
    sigma_s = dpg.get_value("SliderFloat1")
    print(f's:{sigma_s}')


def set_sigma_r(sender):
    sigma_r = dpg.get_value("SliderFloat2")
    print(f'r:{sigma_r}')


with dpg.window(
    label='Sigma & Rolling', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    # dpg.add_text('sigma_s', pos=(5, 20))
    slider_float1 = dpg.add_slider_float(
        tag="SliderFloat1",
        default_value=10,
        max_value=100,
        callback=set_sigma_s
    )

    # dpg.add_text('sigma_r', pos=(5, 40))
    slider_float2 = dpg.add_slider_float(
        tag="SliderFloat2",
        default_value=.1,
        max_value=10,
        callback=set_sigma_r
    )

    dpg.set_item_callback(slider_float1, set_sigma_s)
    dpg.set_item_callback(slider_float2, set_sigma_r)

    def start_rolling():
        global sigma_r, sigma_s, origin_img, origin_image, rgb_channel, rgba_channel, color_change_debug
        image = cv2.ximgproc.rollingGuidanceFilter(
            origin_img, sigmaColor=dpg.get_value("SliderFloat2"), sigmaSpace=dpg.get_value("SliderFloat1"))
        image = np.ravel(image)
        for i in range(0, image_pixels):
            rd_base, im_base = i * rgba_channel, i * rgb_channel
            raw_data[rd_base:rd_base + rgb_channel] = array(
                'f', image[im_base:im_base + rgb_channel]
            )

        color_change_debug += 0.3

        print('finish')

    dpg.add_button(label="start", width=80,
                   pos=(70, 100), callback=start_rolling)

with dpg.window(
    label='Image', pos=(512, 0), tag='Image Win',
    no_move=True, no_close=True, no_collapse=True, no_resize=False, width=1024, height=1024
):
    dpg.add_image("image", show=True, tag='image_data', pos=(10, 30), width=int(
        dpg.get_item_width("image")), height=int(dpg.get_item_height("image")))


dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()


# if paper_name == "":
#
# origin_shape = origin_img.shape
# origin_img = cv2.resize(origin_img, dsize=(
#     512, 512), interpolation=cv2.INTER_LINEAR)
#     img = origin_img
#
#     img = texture(img, sigma=4, turbulence=2)
#
#     img = cv2.resize(img, dsize=(
#         origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_LINEAR)
#
#     plt.imshow(img)
# else:
#     # result = filter_buffer[3] # for fast testing paper texture
#     result = origin_img
#
#     paper_img = cv2.imread(f"{image_dir}/{paper_name}")
#
#     # permutohedralfilter output `float64`(0-255) in filter_buffer[iteration_time], cast back to `uint8`
#     result = result.astype(np.uint8)
#     result = paper.draw_to_paper(result, paper_img)
#
#     plt.imshow(result)
