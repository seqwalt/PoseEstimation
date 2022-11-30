# File system
- PoseEstimation is an object pose estimation project utilizing OpenGL for real-time synthetic data. How to build with make is outlined below.
- src has the source files for the project.
- include has required header files for the projects.
    - stb_image.h is a image loader header from https://github.com/nothings/stb
    - glm is a OpenGL math header library from https://github.com/g-truc/glm.
      Note only the root directory of the header files (also named glm) is
      needed from the glm repo.
    - learnopengl contains useful custom headers adapted from [LearnOpenGL](https://github.com/JoeyDeVries/LearnOpenGL).
- resources has textures/3D-models used in this project.
- opengl_test provides a source file that can be used to test if OpenGL is working properly.
- makefile_template.txt is a template makefile for compiling this project.

# Dependency instructions:
Recommended file structure so provided makefiles work:
```
my_path/OpenGL/
--> deps/
--> learnopengl_projects/ (this repo)
```

## GLAD
- To use GLAD, go to https://glad.dav1d.de/ and select C++, OpenGL, and API gl version
greater than 3.3 (used 4.6 here). Click Generate, and unzip glad folder to
.../OpenGL/deps/glad directory. Also copy glad and KHR include folders to /usr/local/include

## GLFW
- Download the source package from https://www.glfw.org/download.html, unzipping into OpenGl/deps/glfw-3.3.8 directory.
- Follow https://www.glfw.org/docs/latest/compile.html to compile with appropriate OS. The steps for Ubuntu are outlined below.
  ### Build with Ubuntu:
  - Install dependencies for X11 (not using wayland since won't work with nvidia gpu):
    ```
    sudo apt install xorg-dev
    cd deps/glfw-3.3.8
    mkdir build
    ```
  - Generate build files with cmake (make sure to be in deps/glfw-3.3.8 directory):
    ```
    cmake -S . -B build
    ```
  - Compile the library:
    ```
    cd build
    make
    ```

  ### Steps for after build
  - Copy glfw-3.3.8/include/GLFW folder into /usr/local/include
  - In .zshrc or .bashrc, add location of GLFW to PKG_CONFIG_PATH. For example:
  ```export PKG_CONFIG_PATH=my_path/Projects/OpenGL/deps/glfw-3.3.8/build/src:$PKG_CONFIG_PATH```
  - May need to change ownership of /usr/local/include so make can be run without sudo access
    - Check current owner of /usr/local/include. Running ```ls -l``` outputs ```drwxr-xr-x  8 root root 4096 Nov 20 03:46 include```. After the ```chown``` command ```ls -l``` outputs ```drwxr-xr-x  8 my_user my_group 4096 Nov 20 03:46 include```. You can check you group with ```-id -g```.
    ```
    cd /usr/local
    sudo chown -R my_user:my_group include/
    ```
  - You should now be able to compile OpenGL programs :)

## OpenCV
- Note OpenCV core ([opencv](https://github.com/opencv/opencv)) and OpenCV's extra modules ([opencv_contrib](https://github.com/opencv/opencv_contrib)) are required for this project.
- This project was tested with OpenCV 4.6.0.
- To install OpenCV for your platform, see the OpenCV [docs](https://docs.opencv.org/4.6.0/df/d65/tutorial_table_of_content_introduction.html).
- It is recommended to use the stable release of [opencv](https://github.com/opencv/opencv). However for [opencv_contrib](https://github.com/opencv/opencv_contrib), the latest 4.x branch was used with success.
