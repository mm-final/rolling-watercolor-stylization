# Rolling Watercolor Stylization
![alt text](readme_demo_screenshot.png)

We stylize your picture into a watercolor-paint-like style!


### How to Download this?
1. Download the whole project by "Code > Download ZIP" or `git clone` it
2. Make sure you have [Python](https://www.python.org/downloads/)
3. Make sure you `pip install`'d these:
    ```
    pip install numpy 
    pip install dearpygui
    pip install opencv-contrib-python
    ```
    - If `opencv-python` is installed, please uninstall it first. Only one opencv version can be in python.
        ```
        pip uninstall opencv-python 
        ```
        `opencv-contrib-python` will provide all `opencv-python` have with extra modules. This version change won't affect others program you use.
4. Start program by:
    ```
    python main.py
    ```
### Any Tips?
1. **Apply Texture** section paints your result on a paper texture.
    
    Use **select image** button to select your paper image
      - The lower resolution paper image, the coarser the paper looked like in final result. 
      - You can try doing Texture Synthesis to increase image resolution without scaling side effect. Here is a good implement to try: https://github.com/rohitrango/Image-Quilting-for-Texture-Synthesis 
    
    No paper image? Just press **start** without **select image**, we noise generated a paper for you. 
2. Adjusting **sigma_s** and **sigma_r**
 
    **sigma_s** decides how blur the image is. *The bigger, the blurrier the color block*.   
    **sigma_r** decides where is edge of color blocks. *The bigger*, the more differnece between edges is needed to preserve the edge, i.e. *the blurrier at edges*.  
### How Stylization is Done?
Color block effect is done by applying [Rolling Guidance filter](https://www.cse.cuhk.edu.hk/leojia/projects/rollguidance/) to source image.

There are 2 approaches when applying Paper Texture:
1. Have Paper Image:  
   Scaling result image by grayscale value of paper image, then do [gamma correction](https://en.wikipedia.org/wiki/Gamma_correction) to brighten the result back. gamma is $1.2$, fixed.
2. Don't have Paper Image:  
   Generate a noise texture by sampling Standard Normal Distribution to a small area, scale it to result image size, then do same sampling and scaling  multiple time but to a bigger and bigger area, finally add all areas in to one. This generated texture is directly added to result image and  colors' value are clipped back to $[0,255]$.
