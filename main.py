import cv2
import numpy as np
import matplotlib.pyplot as plt
import paper
import dearpygui.dearpygui as dpg
from array import array
import noise_texture

dpg.create_context()
dpg.create_viewport(title='Rolling Watercolor Stylization', width=1280, height=768)


origin_img = None
image_width, image_height, rgb_channel, rgba_channel = None, None, 3, 4
image_pixels = None
raw_data = None

def open_and_set_origin_image(img_path):
    global origin_img, image_width, image_height, rgb_channel, rgba_channel, image_pixels, raw_data
    origin_img = cv2.imread(img_path)
    b, g, r = cv2.split(origin_img) # origin_img preprocessing
    origin_img = cv2.merge([r, g, b])
    origin_img = origin_img / 255.
    origin_img = origin_img.astype(np.float32)

    # reset global variables
    image_width, image_height = origin_img.shape[1], origin_img.shape[0]
    image_pixels = image_height * image_width
    raw_data_size = image_width * image_height * rgba_channel
    raw_data = array('f', [1.] * raw_data_size)

open_and_set_origin_image("images/curious_cat.jpg")


# paper_img
paper_img = None

iteration_time = 4
# b4 s 1 r 0-5
sigma_s = 10
sigma_r = 0.1


texture_format = dpg.mvFormat_Float_rgba
dpg.add_texture_registry(tag="texture_registry", show=False) # use texture_registry for texture removal/addition
dpg.add_raw_texture(
    width=image_width, height=image_height, default_value=raw_data,
    format=texture_format, tag="image", parent="texture_registry"
)


def update_image(new_image):
    global raw_data, image_pixels, rgb_channel, rgba_channel
    for i in range(0, image_pixels):
        rd_base, im_base = i * rgba_channel, i * rgb_channel
        raw_data[rd_base:rd_base + rgb_channel] = array(
            'f', new_image[im_base:im_base + rgb_channel]
        )


width, height = 512, 768
posx, posy = 20, 0
color_change_debug = 0


def set_sigma_s(sender):
    sigma_s = dpg.get_value("SliderFloat1")
    update_log(f's:{sigma_s}')


def set_sigma_r(sender):
    sigma_r = dpg.get_value("SliderFloat2")
    update_log(f'r:{sigma_r}')


with dpg.window(
    label='Sigma & Rolling', width=width, height=height/3, pos=(0, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    dpg.add_text('sigma_s', pos=(posx, 20))
    dpg.add_slider_float(
        tag="SliderFloat1",
        default_value=10,
        max_value=100,
        pos=(posx, 40),
        callback=set_sigma_s
    )

    dpg.add_text('sigma_r', pos=(posx, 60))
    dpg.add_slider_float(
        tag="SliderFloat2",
        default_value=.1,
        max_value=5,
        pos=(posx, 80),
        callback=set_sigma_r
    )

    def start_rolling():
        global sigma_r, sigma_s, origin_img, rgb_channel, rgba_channel, color_change_debug
        update_log("start rolling")

        image = cv2.ximgproc.rollingGuidanceFilter(
            origin_img, sigmaColor=dpg.get_value("SliderFloat2"), sigmaSpace=dpg.get_value("SliderFloat1"))
        image = np.ravel(image)
        update_image(image)

        update_log('finish rolling')

    dpg.add_text('start rolling', pos=(posx, 100))
    dpg.add_button(label="start", width=80,
                   pos=(posx, 120), callback=start_rolling)


    dpg.add_text('change source image', pos=(posx, 150))
    def select_pt(sender, app_data):
        global origin_img, image_width, image_height, raw_data, texture_format
        selections = app_data['selections']
        if selections:
            for fn in selections:
                open_and_set_origin_image(selections[fn])
                dpg.delete_item('image_data')
                dpg.delete_item('image')
                dpg.add_raw_texture(
                    width=image_width, height=image_height, default_value=raw_data,
                    format=texture_format, tag="image", parent="texture_registry"
                )
                dpg.add_image("image", show=True, tag='image_data', pos=(10, 30), width=image_width, 
                    height=image_height, parent="Image Win"
                )
                update_log("load source image successfully")
                break
    
    def cancel_pt(sender, app_data):
        ...

    with dpg.file_dialog(
        directory_selector=False, show=False, callback=select_pt, id='source image selector',
        cancel_callback=cancel_pt, width=700, height=400
    ):
        dpg.add_file_extension('.*')
    dpg.add_button(
        label="select image", width=100, callback=lambda: dpg.show_item("source image selector"),
        pos=(posx, 170),
    )
    

with dpg.window(
    label='Apply Texture', width=width, height=height/3, pos=(0, posy+height/3),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):

    def select_pt(sender, app_data):
        global paper_img
        selections = app_data['selections']
        if selections:
            for fn in selections:
                paper_img = cv2.imread(selections[fn])
                # paper_img = cv2.imread(f"{image_dir}/{paper_name}")
                update_log("load paper texture successfully")
                break

    def cancel_pt(sender, app_data):
        ...

    with dpg.file_dialog(
        directory_selector=False, show=False, callback=select_pt, id='paper selector',
        cancel_callback=cancel_pt, width=700, height=400
    ):
        dpg.add_file_extension('.*')
    dpg.add_button(
        label="select image", width=100, callback=lambda: dpg.show_item("paper selector"),
        pos=(posx, 20),
    )

    def start_texturing(sender):
        global paper_img, origin_img, raw_data
        update_log("start texturing")

        if paper_img is None:
            origin_shape = origin_img.shape
            new_arr = np.array(raw_data)[np.mod(
                np.arange(np.asarray(raw_data).size) + 1, 4) != 0]
            new_arr = new_arr.reshape(origin_shape)
            result = cv2.resize(new_arr, dsize=(
                512, 512), interpolation=cv2.INTER_LINEAR)

            result = noise_texture.texture(result*255, sigma=4, turbulence=4)
            result = result / 255.

            result = cv2.resize(result, dsize=(
                origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_LINEAR)

            image = np.ravel(result)
            update_image(image)

        else:
            origin_shape = origin_img.shape
            new_arr = np.array(raw_data)[np.mod(
                np.arange(np.asarray(raw_data).size) + 1, 4) != 0]
            new_arr = new_arr.reshape(origin_shape)
            # permutohedralfilter output `float64`(0-255) in filter_buffer[iteration_time], cast back to `uint8`
            result = new_arr * 255
            result = result.astype(np.uint8)
            result = paper.draw_to_paper(result, paper_img)

            image = np.ravel(result / 255.)
            update_image(image)

        update_log("finish texturing")

    dpg.add_button(label="start", width=80,
                   pos=(posx, 40), callback=start_texturing)


with dpg.window(
    label='Save Image', width=width, height=height/3, pos=(0, posy+height/3*2),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):

    def save_image(sender):
        global origin_img, raw_data
        data = np.asarray(raw_data) * 255
        dpg.save_image(file="newImage.png",
                       width=origin_img.shape[1], height=origin_img.shape[0], data=data)
        update_log('Result saved as `newImage.png`')

    dpg.add_button(label="save", width=80,
                   pos=(posx, 20), callback=save_image)

    def update_log(log_message): # make log show new message
        dpg.set_value("log", log_message)

    dpg.add_text("Log", pos=(10, 80))
    dpg.add_text('I record want is done!', tag='log', pos=(25, 100))



with dpg.window(
    label='Image', pos=(512, 0), tag='Image Win',
    no_move=True, no_close=True, no_collapse=True, no_resize=False, width=width*2, height=height
):
    dpg.add_image("image", show=True, tag='image_data', pos=(10, 30), width=int(
        dpg.get_item_width("image")), height=int(dpg.get_item_height("image")))


dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
