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
#include <unistd.h>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

// Create Structs
// -----------------------------------------------------
// Struct to hold projection, view and model matrices
struct MatStruct {
  glm::mat4 projection;
  glm::mat4 view;
  glm::mat4 model;
  glm::mat4 est_model_mat; // estimated model matrix
};

// Struct for PnP data
struct PnPdata {
  cv::Mat cameraMat;  // intrinsic camera parameters
  cv::Mat R_matrix;   // rotation matrix
  cv::Mat t_matrix;   // translation matrix
  cv::Mat distCoeffs;  // vector of distortion coefficients (no distortion in sim)
  cv::Mat rvec;              // output rotation vector
  cv::Mat tvec;
  cv::Mat OBJpoints;  // 3D object points used as input to PnP method
  cv::Mat IMGpoints;  // 2D image points also used as input to PnP method
};

// Struct for ORB data
struct ORBdata {
  // initialize keypoints, detector and descriptor for ORB
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> descriptor;

  // matching global variables
  std::vector<cv::KeyPoint> keypointsOld;
  cv::Mat descriptorsOld;
  std::vector<DMatch> matches;
  std::vector<DMatch> good_matches;
  Ptr<DescriptorMatcher> matcher;
};

// Struct to hold ref image data
struct RefImgData {
  ORBdata ORB;
  cv::Mat Features3D;
};

// Struct to hold structs and other stuff
struct MetaData {
  ORBdata ORB;
  PnPdata PnP;
  MatStruct transforms;

  cv::Mat img;
  cv::Mat outimg;
  bool firstFrame;
};
// -----------------------------------------------------

// Global variables
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// Initialize functions
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
bool postProcessingDone(std::future<int>& output);
int drawORBfeatures(MetaData& data);
void estimatePose(const cv::Mat& img, ORBdata& ORB, PnPdata& PnP, MatStruct& transforms, bool firstFrame);
void computeORBfeatures(ORBdata& ORB, const cv::Mat& img);
void computeORBfeatureMatches(ORBdata& ORB, PnPdata& PnP, bool firstFrame);
RefImgData createRefImgData(ORBdata& ORB, cv::Mat& ref_img, const MatStruct& transforms, Shader modelShader, unsigned int texture, unsigned int VAO);
cv::Mat get3Dfeatures(const std::vector<cv::KeyPoint>& keypoints, const MatStruct& mats);
void fromCV2GLM(const cv::Mat& cvmat, glm::mat4* glmmat);
void fromGLM2CV(const glm::mat4& glmmat, cv::Mat* cvmat);

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
    Shader modelShader("shaders/shader.vert", "shaders/shader.frag");
    Shader wireShader("shaders/wireShader.vert", "shaders/wireShader.frag");

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
    unsigned char *data = stbi_load("../resources/textures/kelloggs_cereal.png", &width, &height, &nrChannels, 0);
    if (data){
        // note that the awesomeface.png has transparency and thus an alpha channel, so make sure to tell OpenGL the data type is of GL_RGBA
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else{
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    // Enable z-buffer for depth testing
    glEnable(GL_DEPTH_TEST);

    // tell opengl for each sampler to which texture unit it belongs to (only has to be done once)
    modelShader.use();
    modelShader.setInt("textureSample", 0); // must come after modelShader.use();

    // View matrix to transform to view coords
    // (shift view back so viewer is not at origin -- same as shifting world forward)
    MatStruct transforms;
    glm::mat4 view_mat = glm::mat4(1.0f);
    view_mat = glm::translate(view_mat, glm::vec3(0.0, 0.0, -7.0)); // translate scene toward -z bc OpenGl is a right-handed system
    transforms.view = view_mat;
    // Projection matrix to view world with correct perspective
    glm::mat4 projection_mat = glm::mat4(1.0f);
    float fov_vert = glm::radians(45.0);
    projection_mat = glm::perspective((float)fov_vert, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    transforms.projection = projection_mat;
    modelShader.setMat4("view", view_mat);
    modelShader.setMat4("projection", projection_mat);

    // view and projection matrix are constant
    wireShader.use();
    wireShader.setMat4("view", view_mat);
    wireShader.setMat4("projection", projection_mat);

    // initialize OpenCV image matrix (input and output)
    cv::Mat img(SCR_HEIGHT, SCR_WIDTH, CV_8UC3); // initial rendered image
    cv::Mat outimg(SCR_HEIGHT, SCR_WIDTH, CV_8UC3); // final post-processed image

    // Update storage pixel modes, for post processing
      //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4); // set GL_PACK_ALIGNMENT to 4-byte alignment if img.step is a multiple of 4, else 1-byte assignemnt
      //set length of one complete row in destination data (doesn't need to equal img.cols)
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());

    // initialize keypoints, detector and descriptor for ORB
    ORBdata ORB;
    ORB.detector = cv::ORB::create();
    ORB.descriptor = cv::ORB::create();
    //std::future<int> async_out;

    // matching global variables
    bool firstFrame = false;
    ORB.matcher = DescriptorMatcher::create ( "BruteForce-Hamming" );

    // PnP global variables
    PnPdata PnP;
    PnP.cameraMat = cv::Mat::zeros(3, 3, CV_32F);  // intrinsic camera parameters
    PnP.R_matrix = cv::Mat::zeros(3, 3, CV_32F);   // rotation matrix
    PnP.t_matrix = cv::Mat::zeros(3, 1, CV_32F);   // translation matrix
    PnP.distCoeffs = cv::Mat::zeros(4, 1, CV_32F);  // vector of distortion coefficients (no distortion in sim)
    PnP.rvec = cv::Mat::zeros(3, 1, CV_32F);              // output rotation vector
    PnP.tvec = cv::Mat::zeros(3, 1, CV_32F);              // output translation vector

    // Camera matrix data for PnP algorithm
    PnP.cameraMat.at<float>(0, 0) = ((float)SCR_HEIGHT/2.0)*std::tan(fov_vert/2.0);  //    [ fx   0  cx ]
    PnP.cameraMat.at<float>(1, 1) = ((float)SCR_HEIGHT/2.0)*std::tan(fov_vert/2.0);  //    [  0  fy  cy ]
    PnP.cameraMat.at<float>(0, 2) = (float)SCR_WIDTH/2.0;                            //    [  0   0   1 ]
    PnP.cameraMat.at<float>(1, 2) = (float)SCR_HEIGHT/2.0;
    PnP.cameraMat.at<float>(2, 2) = 1;

    // Initialize meta data struct
    MetaData ARG_DATA;
    ARG_DATA.ORB = ORB;
    ARG_DATA.PnP = PnP;
    ARG_DATA.transforms = transforms;
    ARG_DATA.img = img;
    ARG_DATA.outimg = outimg;
    ARG_DATA.firstFrame = firstFrame;

    // create single reference image
    MatStruct refTransforms = transforms;
    float ref_angle = glm::radians(180.0f);
    glm::vec3 ref_axis = glm::vec3(0.0, 1.0, 0.0);
    glm::mat4 ref_model_mat = glm::mat4(1.0f); // identity
    ref_model_mat = glm::rotate(ref_model_mat, ref_angle, ref_axis);
    refTransforms.model = ref_model_mat;
    RefImgData refImgData = createRefImgData(ORB, img, refTransforms, modelShader, texture, VAO);
    ARG_DATA.PnP.OBJpoints = refImgData.Features3D.clone(); // true clone of opencv matrix (not linked)
    ARG_DATA.ORB.descriptorsOld = refImgData.ORB.descriptors;
    ARG_DATA.transforms.est_model_mat = ref_model_mat;
    glm::mat4 curr_est_model_mat;
    cv::drawKeypoints( img, refImgData.ORB.keypoints, outimg, cv::Scalar(255,100,100), cv::DrawMatchesFlags::DEFAULT );
    cv::flip(outimg, outimg, 0);
    cv::imshow("Reference Image with ORB features", outimg);
    cv::waitKey(0);

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
        float sin_time = std::sin((float)glfwGetTime());
        float cos_time = std::cos((float)glfwGetTime());
        model_mat = glm::translate(model_mat, 0.8f*(sin_time*glm::vec3(sin_time, 1.0f, 0.5f*sin_time) + cos_time*glm::vec3(1.0f, cos_time, 0.5f*sin_time)));
        // model_mat = glm::rotate(model_mat, (float)glfwGetTime()*glm::radians(10.0f), glm::vec3(1.0, 1.0, 0.0));
        model_mat = glm::rotate(model_mat, glm::radians(15.0f), glm::vec3(sin_time, cos_time, 0.0));
        model_mat = glm::rotate(model_mat, (float)glfwGetTime()*glm::radians(10.0f), glm::vec3(0.0, 0.0, 1.0));
        model_mat = glm::rotate(model_mat, ref_angle, glm::vec3(0.0, 1.0, 0.0));

        // Enable the shader program for rendering model
        modelShader.use();
        modelShader.setMat4("model", model_mat); // update uniform
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        // Do asynchronous post-processing
        //if (postProcessingDone(async_out)){
        if (true){
          // display the processed image
          cv::imshow("ORB Features", ARG_DATA.outimg);
          cv::waitKey(1);
          // Convert texture to cv::Mat, and perform orb detection
          glReadPixels(0, 0, ARG_DATA.img.cols, ARG_DATA.img.rows, GL_BGR, GL_UNSIGNED_BYTE, ARG_DATA.img.data);
          ARG_DATA.transforms.model = model_mat;
          curr_est_model_mat = ARG_DATA.transforms.est_model_mat;
          drawORBfeatures(ARG_DATA);
          //async_out = std::async(drawORBfeatures, ARG_DATA); // asynchronously do post-processing to not slow down simulation speed
        }

        // Render wireframe
        wireShader.use();
        glDisable(GL_DEPTH_TEST); // render completely in front of cereal box
        wireShader.setMat4("model", curr_est_model_mat);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // wireframe
        glBindVertexArray(wireVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glEnable(GL_DEPTH_TEST); // re-enable depth testing

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window); // See Double Buffer note in LearnOpenGL book
        glfwPollEvents();

        // std::cout << glm::to_string(est_model_mat) << std::endl;
        // usleep(20000000);

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

bool postProcessingDone(std::future<int>& output){
    // true if async function is complete, else false
    return output.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

int drawORBfeatures(MetaData& data)
{
  estimatePose(data.img, data.ORB, data.PnP, data.transforms, data.firstFrame); // update structs

  cv::drawKeypoints( data.img, data.ORB.keypoints, data.outimg, cv::Scalar(255,100,100), cv::DrawMatchesFlags::DEFAULT );
  cv::flip(data.outimg, data.outimg, 0);

  // matching code
  data.firstFrame = true;
  data.ORB.keypointsOld = data.ORB.keypoints;
  data.ORB.descriptorsOld = data.ORB.descriptors;

  return 0;
}

void estimatePose(const cv::Mat& img, ORBdata& ORB, PnPdata& PnP, MatStruct& transforms, bool firstFrame)
{
  // NOTE: this function uses pass-by-reference to update ORB, PnP and transforms structs

  // 1) Reference image --> orb features & descriptors of ref img --> 3D coords of features
  // 2) Render first sim img --> orb features & descriptors of sim img --> feature matching
  //    of image points between sim and ref features --> associate 3D object points
  //    (w.r.t. object frame) from ref img with image points in sim img.
  // 3) Solve EPnP problem, which inputs the 3D object points and 2D image points,
  //    and outputs the rotation and translation vectors that transform a 3D object point
  //    to the camera coordinate frame.
  computeORBfeatures(ORB, img);

  computeORBfeatureMatches(ORB, PnP, firstFrame);

  // Run PnP algorithm
  // ---------------------------------------------------------
  cv::solvePnP(PnP.OBJpoints, PnP.IMGpoints, PnP.cameraMat, PnP.distCoeffs, PnP.rvec, PnP.tvec, false, SOLVEPNP_EPNP);
  cv::Rodrigues(PnP.rvec, PnP.R_matrix);   // converts Rotation Vector to Matrix. Rotation from obj frame to view frame
  PnP.t_matrix = PnP.tvec;                 // set translation matrix. Translation from obj frame to view frame
  // ---------------------------------------------------------

  // Estimate pose
  // ---------------------------------------------------------
  // Convert to homography matrix called est_model_mat_cv,
  cv::Mat est_model_mat_cv = PnP.R_matrix;
  cv::hconcat(est_model_mat_cv, PnP.t_matrix, est_model_mat_cv); // concatonate the translation vector
  cv::Mat bot_row = (cv::Mat_<float>(1, 4) << 0, 0, 0, 1);
  est_model_mat_cv.push_back(bot_row); // add on the bottom row

  cv::Mat view_mat_cv;
  fromGLM2CV(transforms.view, &view_mat_cv);

  est_model_mat_cv = est_model_mat_cv * (-view_mat_cv);

  fromCV2GLM(est_model_mat_cv, &transforms.est_model_mat); // updates transforms.est_model_mat value
  // ---------------------------------------------------------
}

void computeORBfeatures(ORBdata& ORB, const cv::Mat& img)
{
  // Oriented FAST
  ORB.detector->detect(img, ORB.keypoints); // keypoints is filled with data
  // Rotated BRIEF
  ORB.descriptor->compute(img, ORB.keypoints, ORB.descriptors); // descriptors is filled with data
}

// This function computes matches, and for the "good" matches, it performs the 2D-3D
// point correspondenses
void computeORBfeatureMatches(ORBdata& ORB, PnPdata& PnP, bool firstFrame)
{
  if (firstFrame) {
    ORB.matcher->match ( ORB.descriptors, ORB.descriptorsOld, ORB.matches);
    double min_dist=10000, max_dist=0;
    for ( int i = 0; i < ORB.descriptorsOld.rows; i++ ) {
      double dist = ORB.matches[i].distance;
      if ( dist < min_dist ) min_dist = dist;
      if ( dist > max_dist ) max_dist = dist;
    }
    bool short_hamming_dist;
    int ind_old; // index of feature in old image
    int ind_new; // index of feature in new image
    int good_ind = 0;
    cv::Mat dummy_row = cv::Mat::zeros(1, 2, CV_32F); // dummy row for IMGpoints matrix
    for ( int i = 0; i < ORB.descriptorsOld.rows; i++ ) {
        short_hamming_dist = (ORB.matches[i].distance <= std::max( 2*min_dist, 30.0 )); // Is the hamming distance between features short enough?
        //short_euclid_dist = (); // Is the euclidean distance between features short enough?
        if ( short_hamming_dist )
        {
          ORB.good_matches.push_back ( ORB.matches[i] ); // save good matches information

          // Find 2D keypoints and corresponding 3D object points
          // ---------------------------------------------------------
          // save 3D object points into Nx3 cv::Mat
          ind_old = ORB.good_matches[i].trainIdx; // index of old image keypoints

          // TODO: Is the following line correct?

          PnP.OBJpoints.push_back(PnP.OBJpoints.row(ind_old)); // each row of oldCoords3D is a 3D object coordinate

          // save 2D img coordinates (in newer image) into Nx2 cv::Mat
          ind_new = ORB.good_matches[i].queryIdx; // index of new image keypoints
          PnP.IMGpoints.push_back(dummy_row);
          PnP.IMGpoints.at<float>(good_ind, 0) = ORB.keypoints[ind_new].pt.x; // assign values in dummy row
          PnP.IMGpoints.at<float>(good_ind, 1) = ORB.keypoints[ind_new].pt.y;
          // ---------------------------------------------------------
          good_ind += 1;
        }
    }
  }
}

RefImgData createRefImgData(ORBdata& ORB, cv::Mat& ref_img, const MatStruct& transforms, Shader modelShader, unsigned int texture, unsigned int VAO)
{
   // Create single reference image
   // -----------------------------
  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear depth and color buffers
  // bind textures on corresponding texture units
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  // Enable the shader program for rendering model
  modelShader.use();
  modelShader.setMat4("model", transforms.model); // update uniform
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glBindVertexArray(VAO);
  glDrawArrays(GL_TRIANGLES, 0, 36);
  // Convert texture to cv::Mat, and perform orb detection
  glReadPixels(0, 0, ref_img.cols, ref_img.rows, GL_BGR, GL_UNSIGNED_BYTE, ref_img.data);
  // -----------------------------

  // Compute 3D feature locations in model space
  // -----------------------------
  computeORBfeatures(ORB, ref_img); // keypoints variable and descriptors variable now are loaded with data
  RefImgData refImgData;
  refImgData.ORB = ORB;
  refImgData.Features3D = get3Dfeatures(ORB.keypoints, transforms);
  // -----------------------------

  return refImgData;
}

// Extract 3D object coordinates from ORB features in world frame
// https://stackoverflow.com/questions/25687213/how-does-gl-position-becomes-a-x-y-position-in-the-window
// https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluUnProject.xml
// https://www.khronos.org/opengl/wiki/GluProject_and_gluUnProject_code
cv::Mat get3Dfeatures(const std::vector<cv::KeyPoint>& keypoints, const MatStruct& mats)
{
  GLint viewport[4];
  glGetIntegerv(GL_VIEWPORT, viewport); // retrieves viewport values (x, y, width, height)

  // Convert keypoints into NDC normalized device coordinates (x,y,depth). Note all values are between 0 and 1.
  glm::vec4 NDCpos;
  float depth;
  glm::mat4 PVM_inv = glm::inverse(mats.projection * mats.view * mats.model);
  cv::Mat objCoords = cv::Mat(keypoints.size(), 3, CV_32F);  // rows, columns, type (using floats here)

  for (int i = 0; i < keypoints.size(); i++){
    // Transformation of normalized coordinates between -1 and 1
    NDCpos[0] = (keypoints[i].pt.x/SCR_WIDTH)*2 - 1;
    NDCpos[1] = (keypoints[i].pt.y/SCR_HEIGHT)*2 - 1;
    glReadPixels(keypoints[i].pt.x, keypoints[i].pt.y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
    NDCpos[2] = 2*depth - 1;
    NDCpos[3] = 1.0f;

    glm::vec4 out = PVM_inv * NDCpos; // unnormalized object coordinates
    out[3]=1.0/out[3];
    objCoords.at<float>(i,0) = out[0]*out[3];
    objCoords.at<float>(i,1) = out[1]*out[3];
    objCoords.at<float>(i,2) = out[2]*out[3];
  }

  return objCoords;
}

// convert from cv::Mat to glm::mat4
// https://stackoverflow.com/questions/44409443/how-a-cvmat-translate-from-to-a-glmmat4
void fromCV2GLM(const cv::Mat& cvmat, glm::mat4* glmmat)
{
    if (cvmat.cols != 4 || cvmat.rows != 4 || cvmat.type() != CV_32F) {
        std::cout << "Matrix conversion error!" << std::endl;
        return;
    }
    memcpy(glm::value_ptr(*glmmat), cvmat.data, 16 * sizeof(float));
}

// convert from glm::mat4 to cv::Mat
void fromGLM2CV(const glm::mat4& glmmat, cv::Mat* cvmat)
{
    if (cvmat->cols != 4 || cvmat->rows != 4) {
        (*cvmat) = cv::Mat(4, 4, CV_32F);
    }
    memcpy(cvmat->data, glm::value_ptr(glmmat), 16 * sizeof(float));
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
