# By Sequoyah Walters, Sriram Ashokkumar, and Jingxin Du


## Project Presentation
### [Slides](https://docs.google.com/presentation/d/1wN2tkIdrWc6zNNb7Fr0up25WB-uktFyzhzcq5vQqLLs/edit?usp=sharing)
### Video


# Problem
Many applications require the knowledge of the position and orientation (pose) of an object sometimes even
predicting the pose of a moving object. This problem is not necessarily novel and numerous solutions exist.
However, modern approaches to this issue use computationally intensive neural network (NN) methods, restricting 
usage of applications with low size, weight and power (SWaP). We estimated the object pose
with similar performance to NN methods on a simple problem domain for usage by low-SWaP systems. Sim-
plifying assumptions include knowledge of the object 3D model, a non-cluttered scene and only considering
the pose of a single object.

# Motivation
We are interested in this problem as we want to predict the pose of an object using low SWaP systems. Current solutions to this problem require extensive computation and would not be feasible in a low SWaP system. There are several applications in which this approach would be beneficial, including quick obstacle avoidance and fast object trajectory estimation. 

Low SWaP systems include quadcopters where there is a restriction on computing power due to their small size. An example where our solution could be used would be \cite{falanga_aggressive_2017}, where the goal is for a quadcopter to fly through a narrow gap in an object as shown in Figure INSERT QUADCOMPTER PICTURE. In this situation, there is a limitation on computational power as the quadcopter has to be light and the algorithm also needs to be fast due to high-speed flight.

# Current State-of-the-Art
Not sure if we need this


# Approach
<!-- This work aims to re-implement and improve a classical pose estimation approach for a unique setting. By
considering the application of low-SWaP systems, our algorithm must be extremely light-weight and efficient
in order to obtain good performance. Existing approaches do not consider the use of low-SWaP systems, so
we aim to design an algorithm that can perform better for these types of systems. -->
We predicted the pose of a single object with a known 3D model. ORB features
will be detected in the image, and compared to a reference image for feature matching. Using EPnP, the
popular 2D/3D perspective-n-point method proposed in [13], we can quickly generate pose estimations from
point correspondences.

## 3D Object Model - OpenGL
OpenGL was used with C++ to create a very simple rendering engine. It currently
can display in 3D a box object, as shown in Figure 4. While this OpenGl renderer is very simple, it allows
us full control to display the wire-mesh of the model, which is useful when estimating pose with our
algorithm. 

<!-- Additionally, as long as we can render a model that has visually rich regions (such as the cereal
box), our algorithm should be able to detect features properly. -->

![Cereal Box](/PoseEstimation/docs/assets/logo.png)


## ORB Feature Extraction and Matching
ORB features [12] are extracted from each reference image and stored. These 2D features have known 3D counterparts. 
Next, ORB feature detection is computed over the new image, and feature matching is done between this 
new image and a reference image. Since the features in the reference image have known 3D values, 
this information is used for pose estimation described in the next section.

## Pose Estimation
For pose estimation, we used Efficient Perspective-n-Point(EPnP) as shown in Figure 3. The EPnP
approach takes in the features in the image that coordinate to the 3D points as a weighted sum of four
virtual control points. This is less computationally heavy as it allows us to reduces the problem to estimating
the position of these control points in the camera reference frame. The solution is an accurate non-iterative
O(n) solution to the PnP problem. 


## Evaluation
Include information from evaluation



