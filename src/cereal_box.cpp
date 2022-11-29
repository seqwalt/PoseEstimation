#define STB_IMAGE_IMPLEMENTATION
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <learnopengl/shader_m.h>

#include <iostream>
#include <future>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
int drawORBfeatures();
void computeORBfeatures();

// settings
const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 900;

// post processing
// -----------------------------------------------------
// initialize OpenCV image matrix
cv::Mat img(SCR_HEIGHT, SCR_WIDTH, CV_8UC3); // initial rendered image
cv::Mat outimg(SCR_HEIGHT, SCR_WIDTH, CV_8UC3); // final post-processed image
// initialize keypoints, detector and descriptor for ORB
std::vector<cv::KeyPoint> keypoints;
cv::Mat descriptors;
cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
bool postProcessingDone = true;
std::future<int> async_out;
// -----------------------------------------------------

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // build and compile our shader program
    // ------------------------------------
    Shader modelShader("shader.vert", "shader.frag");
    Shader wireShader("wireShader.vert", "wireShader.frag");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------- CUBE!!!!! --------------------------------------
    float hL = 1.0f; // half length
    float hH = 1.2934f; // half height
    float hW = 0.41148985f; // half width

    // texture coordinates from png pixel locations
    float x1 = 25.0f/2048.0f; // column 1 texture values (x-axis. left of png)
    float x2 = 319.0f/2048.0f;
    float x3 = 1024.0f/2048.0f;
    float x4 = 1315.0f/2048.0f;
    float x5 = 2038.0f/2048.0f;
    float y1 = (2048.0f - 1767.0f)/2048.0f; // row 1 texture values (y-axis. bottom of png)
    float y2 = (2048.0f - 1467.0f)/2048.0f; // normalize an make bottom left origin, and top right (1,1)
    float y3 = (2048.0f - 534.0f)/2048.0f;
    float y4 = (2048.0f - 237.0f)/2048.0f;

    float vertices[] = {
      // position     // texture coordinates

      // bottom side of cereal box
      -hL, -hH, -hW,  x2, y1,
       hL, -hH, -hW,  x3, y1,
       hL, -hH,  hW,  x3, y2,
       hL, -hH,  hW,  x3, y2,
      -hL, -hH,  hW,  x2, y2,
      -hL, -hH, -hW,  x2, y1,
      // top side of cereal box
      -hL,  hH, -hW,  x2, y4,
       hL,  hH, -hW,  x3, y4,
       hL,  hH,  hW,  x3, y3,
       hL,  hH,  hW,  x3, y3,
      -hL,  hH,  hW,  x2, y3,
      -hL,  hH, -hW,  x2, y4,
      // left side of cereal box
      -hL,  hH,  hW,  x2, y3,
      -hL, -hH,  hW,  x2, y2,
      -hL, -hH, -hW,  x1, y2,
      -hL, -hH, -hW,  x1, y2,
      -hL,  hH, -hW,  x1, y3,
      -hL,  hH,  hW,  x2, y3,
      // right side of cereal box
       hL,  hH,  hW,  x3, y3,
       hL, -hH,  hW,  x3, y2,
       hL, -hH, -hW,  x4, y2,
       hL, -hH, -hW,  x4, y2,
       hL,  hH, -hW,  x4, y3,
       hL,  hH,  hW,  x3, y3,
       // back side of cereal box
      -hL, -hH, -hW,  x5, y2,
       hL, -hH, -hW,  x4, y2,
       hL,  hH, -hW,  x4, y3,
       hL,  hH, -hW,  x4, y3,
      -hL,  hH, -hW,  x5, y3,
      -hL, -hH, -hW,  x5, y2,
      // front side of cereal box (facing camera)
      -hL, -hH,  hW,  x2, y2,
       hL, -hH,  hW,  x3, y2,
       hL,  hH,  hW,  x3, y3,
       hL,  hH,  hW,  x3, y3,
      -hL,  hH,  hW,  x2, y3,
      -hL, -hH,  hW,  x2, y2
    };

    // Model objects
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO); // only want 1 vertex array object generated
    glGenBuffers(1, &VBO);
    // Bind VAO
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO); // a VBO has buffer type GL_ARRAY_BUFFER
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); // copy the vertex data into the buffer memory
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0); // location of aPos attribute is 0
    // texture attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1); // location of aTexCoord attribute is 1

    // Wireframe objects
    unsigned int wireVAO;
    glGenVertexArrays(1, &wireVAO);
    glBindVertexArray(wireVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO); // a VBO has buffer type GL_ARRAY_BUFFER
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); // copy the vertex data into the buffer memory
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0); // location of aPos attribute is 0

    // load and create a texture
    // -------------------------
    unsigned int texture;
    // cereal box texture
    // ---------
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR); //GL_NEAREST_MIPMAP_LINEAR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    unsigned char *data = stbi_load("kelloggs_cereal.png", &width, &height, &nrChannels, 0);
    if (data){
        // note that the awesomeface.png has transparency and thus an alpha channel, so make sure to tell OpenGL the data type is of GL_RGBA
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else{
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    // tell opengl for each sampler to which texture unit it belongs to (only has to be done once)
    modelShader.use();
    modelShader.setInt("textureSample", 0); // must come after modelShader.use();

    // View matrix to transform to view coords
    // (shift view back so viewer is not at origin -- same as shifting world forward)
    glm::mat4 view_mat = glm::mat4(1.0f);
    view_mat = glm::translate(view_mat, glm::vec3(0.0, 0.0, -6.0)); // translate scene toward -z bc OpenGl is a right-handed system
    // Projection matrix to view world with correct perspective
    glm::mat4 projection_mat = glm::mat4(1.0f);
    projection_mat = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    modelShader.setMat4("view", view_mat);
    modelShader.setMat4("projection", projection_mat);

    // view and projection matrix are constant
    wireShader.use();
    wireShader.setMat4("view", view_mat);
    wireShader.setMat4("projection", projection_mat);

    // Update storage pixel modes, for post processing
      //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4); // set GL_PACK_ALIGNMENT to 4-byte alignment if img.step is a multiple of 4, else 1-byte assignemnt
      //set length of one complete row in destination data (doesn't need to equal img.cols)
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());

    // post processing
    bool firstORBcompute = true;

    // Enable z-buffer for depth testing
    glEnable(GL_DEPTH_TEST);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // render
        // -----
        // background color
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear depth and color buffers

        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);

        // Model matrix to transform to world coords
        glm::mat4 model_mat = glm::mat4(1.0f); // identity
        model_mat = glm::rotate(model_mat, (float)glfwGetTime() * glm::radians(50.0f), glm::vec3(0.3, 1.0, 0.0));

        // Enable the shader program for rendering model
        modelShader.use();
        modelShader.setMat4("model", model_mat); // update uniform
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        // Do asynchronous post-processing
        if (postProcessingDone){
          postProcessingDone = false;
          // Convert texture to cv::Mat, and perform orb detection
          glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, img.data);
          async_out = std::async(drawORBfeatures); // asynchronously do post-processing to not slow down simulation speed
        }

        // Render wireframe
        wireShader.use();
        glDisable(GL_DEPTH_TEST); // render completely in front of cereal box
        wireShader.setMat4("model", model_mat); // TEMPORARY --> eventually will be using estimated model matrix here, not ground truth
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // wireframe
        glBindVertexArray(wireVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glEnable(GL_DEPTH_TEST); // re-enable depth testing

        // TODO:
        // Implement map from pixel location to 3D point in model coordinates (inverse of
        // projection_mat*view_mat*model_mat ?). This will allow
        // for generation of reference images with cooresponding 3D cooridnates for use with EPnP.

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window); // See Double Buffer note in LearnOpenGL book
        glfwPollEvents();
    }

    // de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    modelShader.end(); // delete shader program

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

int drawORBfeatures()
{
  computeORBfeatures();
  cv::drawKeypoints( img, keypoints, outimg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
  cv::flip(outimg, outimg, 0);
  cv::imshow("ORB Features",outimg);
  cv::waitKey(1);
  postProcessingDone = true;
  return 0;
}

void computeORBfeatures()
{
  // Oriented FAST
  detector->detect(img, keypoints);
  // Rotated BRIEF
  descriptor->compute(img, keypoints, descriptors);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
