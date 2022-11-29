# Check if using Mac (Darwin OS) or Linux
current_OS := $(shell uname -s)
ifeq ($(current_OS),Linux)
	OPENGL_DIR=my/linux/directory/to/OpenGL
	CFLAGS=`pkg-config --cflags glfw3 opencv4`
	LIBS=`pkg-config --static --libs glfw3 opencv4`
endif
ifeq ($(current_OS),Darwin)
	OPENGL_DIR=my/mac/directory/to/OpenGL
	CFLAGS=`pkg-config --cflags glfw3 opencv`
	LIBS=`pkg-config --static --libs glfw3 opencv`
endif

GLAD_SRC_DIR=$(OPENGL_DIR)/deps/glad/src
GLFW_PATH=$(OPENGL_DIR)/deps/glfw-3.3.8/build/src
INCLUDE_PATH=$(OPENGL_DIR)/learnopengl_projects/include
CPP_VER=17
MAIN_NAME=cereal_box

# target: prerequisites
main_executable: $(GLAD_SRC_DIR)/glad.o # note: @ tells make not to print the command being executed
	@echo Generating $(MAIN_NAME) executable
	@g++ -std=c++$(CPP_VER) $(CFLAGS) -o $(MAIN_NAME) $(GLAD_SRC_DIR)/glad.o $(MAIN_NAME).cpp $(LIBS) -L $(GLFW_PATH) -I $(INCLUDE_PATH)

$(GLAD_SRC_DIR)/glad.o: # executes if glad.o not in directory
	@echo Generating GLAD object file
	@gcc -c -o $(GLAD_SRC_DIR)/glad.o $(GLAD_SRC_DIR)/glad.c

clean:
	@echo Removing GLAD object file
	@rm $(GLAD_SRC_DIR)/glad.o
	@echo Removing $(MAIN_NAME) executable
	@rm $(MAIN_NAME)

.PHONY: clean
