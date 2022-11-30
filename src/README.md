# File system
- cereal_box.cpp is the main file that runs the OpenGL renderer and does pose estimation.
- shaders/ contains the various vertex and fragment shaders used by cereal_box.cpp.

# Use makefile to automate code compilation
1. copy makefile_template.txt to this src directory and rename the copy "makefile".
3. update paths in makefile to reflect your system.
4. run ```make``` to generate the executable. run ```./cereal_box``` to run the executable.
5. run ```make debug``` to generate the executable and gdb debugging info, which can be used with the gdb command line tool.
5. run ```make clean``` to remove glad.o object file from glad/src and the executable and debug folder from the current directory
