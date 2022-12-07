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

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
int drawORBfeatures();
void computeORBfeatures();
void computeORBfeatureMatches();
void estimatePose();
cv::Mat get3Dfeatures(glm::mat4 projection_mat, glm::mat4 view_mat, glm::mat4 model_mat);
void fromCV2GLM(const cv::Mat& cvmat, glm::mat4* glmmat);
void fromGLM2CV(const glm::mat4& glmmat, cv::Mat* cvmat);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// post processing
// -----------------------------------------------------
// initialize OpenCV image matrix
cv::Mat img(SCR_HEIGHT, SCR_WIDTH, CV_8UC3); // initial rendered image
cv::Mat outimg(SCR_HEIGHT, SCR_WIDTH, CV_8UC3); // final post-processed image

// Create struct that holds ORB descriptors and associated 3D vector for each detected feature
// Used for reference image creation, and EPnP
struct OrbData3D {
  cv::Mat OrbDescriptors;
  cv::Mat Features3D;
};
// initialize function
OrbData3D createRefImgData(Shader modelShader, unsigned int texture, unsigned int VAO, glm::mat4 projection_mat, glm::mat4 view_mat, glm::mat4 ref_model_mat);
float ref_angle = glm::radians(180.0f);
glm::vec3 ref_axis = glm::vec3(0.0, 1.0, 0.0);

// initialize keypoints, detector and descriptor for ORB
std::vector<cv::KeyPoint> keypoints;
cv::Mat descriptors;
cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
bool postProcessingDone = true;
std::future<int> async_out;

// matching global variables
bool firstFrame = false;
std::vector<cv::KeyPoint> keypointsOld;
cv::Mat descriptorsOld;
std::vector<DMatch> matches;
std::vector<DMatch> good_matches;
Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

// Pose global variables
glm::mat4 est_model_mat; // estimated model matrix
cv::Mat Pose, prevPose;
cv::Mat Coords3D, oldCoords3D;
glm::mat4 view_mat;

// PnP globa variables
float fov_vert = glm::radians(45.0);
cv::Mat cameraMat = cv::Mat::zeros(3, 3, CV_32F);  // intrinsic camera parameters
cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_32F);   // rotation matrix
cv::Mat t_matrix = cv::Mat::zeros(3, 1, CV_32F);   // translation matrix
const cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_32F);  // vector of distortion coefficients (no distortion in sim)
cv::Mat rvec = cv::Mat::zeros(3, 1, CV_32F);              // output rotation vector
cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32F);              // output translation vector

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

    // tell opengl for each sampler to which texture unit it belongs to (only has to be done once)
    modelShader.use();
    modelShader.setInt("textureSample", 0); // must come after modelShader.use();

    // View matrix to transform to view coords
    // (shift view back so viewer is not at origin -- same as shifting world forward)
    view_mat = glm::mat4(1.0f);
    view_mat = glm::translate(view_mat, glm::vec3(0.0, 0.0, -7.0)); // translate scene toward -z bc OpenGl is a right-handed system
    // Projection matrix to view world with correct perspective
    glm::mat4 projection_mat = glm::mat4(1.0f);
    projection_mat = glm::perspective((float)fov_vert, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
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

    // create single reference image
    glm::mat4 ref_model_mat = glm::mat4(1.0f); // identity
    ref_model_mat = glm::rotate(ref_model_mat, ref_angle, ref_axis);
    OrbData3D refImgData = createRefImgData(modelShader, texture, VAO, projection_mat, view_mat, ref_model_mat);
    oldCoords3D = refImgData.Features3D.clone(); // true clone of opencv matrix (not linked)
    cv::drawKeypoints( img, keypoints, outimg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    cv::flip(outimg, outimg, 0);
    cv::imshow("Reference Image with ORB features", outimg);
    cv::waitKey(0);

    // Camera matrix data for PnP algorithm
    cameraMat.at<float>(0, 0) = ((float)SCR_HEIGHT/2.0)*std::tan(fov_vert/2.0);  //    [ fx   0  cx ]
    cameraMat.at<float>(1, 1) = ((float)SCR_HEIGHT/2.0)*std::tan(fov_vert/2.0);  //    [  0  fy  cy ]
    cameraMat.at<float>(0, 2) = (float)SCR_WIDTH/2.0;                            //    [  0   0   1 ]
    cameraMat.at<float>(1, 2) = (float)SCR_HEIGHT/2.0;
    cameraMat.at<float>(2, 2) = 1;

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
        if (postProcessingDone){
          // display the processed image
          cv::imshow("ORB Features",outimg);
          cv::waitKey(1);
          postProcessingDone = false;
          // Convert texture to cv::Mat, and perform orb detection
          glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, img.data);
          async_out = std::async(drawORBfeatures); // asynchronously do post-processing to not slow down simulation speed
        }

        // Render wireframe
        wireShader.use();
        glDisable(GL_DEPTH_TEST); // render completely in front of cereal box
        wireShader.setMat4("model", est_model_mat);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // wireframe
        glBindVertexArray(wireVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glEnable(GL_DEPTH_TEST); // re-enable depth testing

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
  estimatePose();

  cv::drawKeypoints( img, keypoints, outimg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
  cv::flip(outimg, outimg, 0);

  postProcessingDone = true;

  // matching code
  firstFrame = true;
  keypointsOld = keypoints;
  descriptorsOld = descriptors;

  return 0;
}

void estimatePose()
{
  // 1) Reference image --> orb features & descriptors of ref img --> 3D coords of features
  // 2) Render first sim img --> orb features & descriptors of sim img --> feature matching
  //    of image points between sim and ref features --> associate 3D object points
  //    (w.r.t. object frame) from ref img with image points in sim img.
  // 3) Solve EPnP problem, which inputs the 3D object points and 2D image points,
  //    and outputs the rotation and translation vectors that transform a 3D object point
  //    to the camera coordinate frame.
  computeORBfeatures();

  computeORBfeatureMatches();
}

void computeORBfeatures()
{
  // Oriented FAST
  detector->detect(img, keypoints); // keypoints is filled with data
  // Rotated BRIEF
  descriptor->compute(img, keypoints, descriptors); // descriptors is filled with data
}

void computeORBfeatureMatches()
{
  if (firstFrame) {
    matcher->match ( descriptors, descriptorsOld, matches);
    double min_dist=10000, max_dist=0;
    for ( int i = 0; i < descriptorsOld.rows; i++ ) {
      double dist = matches[i].distance;
      if ( dist < min_dist ) min_dist = dist;
      if ( dist > max_dist ) max_dist = dist;
    }
    bool short_hamming_dist;
    int ind_old; // index of feature in old image
    int ind_new; // index of feature in new image
    int good_ind = 0;
    cv::Mat OBJpoints; // 3D object points used as input to PnP method
    cv::Mat IMGpoints; // 2D image points also used as input to PnP method
    cv::Mat dummy_row = cv::Mat::zeros(1, 2, CV_32F); // dummy row for IMGpoints matrix
    for ( int i = 0; i < descriptorsOld.rows; i++ ) {
        short_hamming_dist = (matches[i].distance <= std::max( 2*min_dist, 30.0 )); // Is the hamming distance between features short enough?
        //short_euclid_dist = (); // Is the euclidean distance between features short enough?
        if ( short_hamming_dist )
        {
          good_matches.push_back ( matches[i] ); // save good matches information

          // Find 2D keypoints and corresponding 3D object points
          // ---------------------------------------------------------
          // save 3D object points into Nx3 cv::Mat
          ind_old = good_matches[i].trainIdx; // index of old image keypoints
          OBJpoints.push_back(oldCoords3D.row(ind_old)); // each row of oldCoords3D is a 3D object coordinate

          // save 2D img coordinates (in newer image) into Nx2 cv::Mat
          ind_new = good_matches[i].queryIdx; // index of new image keypoints
          IMGpoints.push_back(dummy_row);
          IMGpoints.at<float>(good_ind, 0) = keypoints[ind_new].pt.x; // assign values in dummy row
          IMGpoints.at<float>(good_ind, 1) = keypoints[ind_new].pt.y;
          // ---------------------------------------------------------
          good_ind += 1;
        }
    }
    // Run PnP algorithm
    // ---------------------------------------------------------
    cv::solvePnP(OBJpoints, IMGpoints, cameraMat, distCoeffs, rvec, tvec, false, SOLVEPNP_EPNP);
    cv::Rodrigues(rvec, R_matrix);   // converts Rotation Vector to Matrix. Rotation from obj frame to view frame
    t_matrix = tvec;                 // set translation matrix. Translation from obj frame to view frame
    // ---------------------------------------------------------

    // Estimate pose
    // ---------------------------------------------------------
    // Convert to homography matrix called est_model_mat_cv,
    cv::Mat est_model_mat_cv = R_matrix;
    cv::hconcat(est_model_mat_cv, t_matrix, est_model_mat_cv); // concatonate the translation vector
    cv::Mat bot_row = (cv::Mat_<float>(1, 4) << 0, 0, 0, 1);
    est_model_mat_cv.push_back(bot_row); // add on the bottom row

    cv::Mat view_mat_cv;
    fromGLM2CV(view_mat, &view_mat_cv);

    est_model_mat_cv = est_model_mat_cv * (-view_mat_cv);

    fromCV2GLM(est_model_mat_cv, &est_model_mat);
    // ---------------------------------------------------------
  }
}

OrbData3D createRefImgData(Shader modelShader, unsigned int texture, unsigned int VAO, glm::mat4 projection_mat, glm::mat4 view_mat, glm::mat4 model_mat)
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
  modelShader.setMat4("model", model_mat); // update uniform
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glBindVertexArray(VAO);
  glDrawArrays(GL_TRIANGLES, 0, 36);
  // Convert texture to cv::Mat, and perform orb detection
  glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, img.data);
  // -----------------------------

  // Compute 3D feature locations in model space
  // -----------------------------
  computeORBfeatures(); // keypoints variable and descriptors variable now are loaded with data
  OrbData3D refImgData;
  refImgData.OrbDescriptors = descriptors;
  refImgData.Features3D = get3Dfeatures(projection_mat, view_mat, model_mat);
  // -----------------------------

  return refImgData;
}

// Extract 3D object coordinates from ORB features in world frame
// https://stackoverflow.com/questions/25687213/how-does-gl-position-becomes-a-x-y-position-in-the-window
// https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluUnProject.xml
// https://www.khronos.org/opengl/wiki/GluProject_and_gluUnProject_code
cv::Mat get3Dfeatures(glm::mat4 projection_mat, glm::mat4 view_mat, glm::mat4 model_mat)
{
  GLint viewport[4];
  glGetIntegerv(GL_VIEWPORT, viewport); // retrieves viewport values (x, y, width, height)

  // Convert keypoints into NDC normalized device coordinates (x,y,depth). Note all values are between 0 and 1.
  glm::vec4 NDCpos;
  float depth;
  glm::mat4 PVM_inv = glm::inverse(projection_mat * view_mat * model_mat);
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
