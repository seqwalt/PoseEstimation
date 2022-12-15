# Pose Estimation
### By Sequoyah Walters(NetID: snwalters2), Sriram Ashokkumar(NetID: ashokkumar2), and Jingxin Du(NetID:jdu86)


# Resources
### [Project Presentation Slides](https://docs.google.com/presentation/d/1wN2tkIdrWc6zNNb7Fr0up25WB-uktFyzhzcq5vQqLLs/edit?usp=sharing)
### [Project Presentation Video](https://uwprod-my.sharepoint.com/:v:/g/personal/snwalters2_wisc_edu1/EUkXcHxw2dlCgaERJGYHZI4BxNixHhj5xG-2ZKZ3xNmBFQ?e=WHZoqO)
### [Project Proposal](https://drive.google.com/file/d/1hC8oRceYPUFodSvo5DaotZEf3OexQ76n/view?usp=share_link)
### [Project Mid-term Report](https://drive.google.com/file/d/178fejKkhdA7yCzVx__yCsbTKTNsXARnO/view?usp=share_link)
### [Project Source Codes](https://github.com/seqwalt/PoseEstimation)

# Problem
Many applications require the knowledge of the position and orientation (pose) of an object sometimes even
predicting the pose of a moving object. This problem is not necessarily novel and numerous solutions exist.
However, modern approaches to this issue use computationally intensive neural network (NN) methods, restricting 
usage of applications with low size, weight and power (SWaP). We estimated the object pose
with similar performance to NN methods on a simple problem domain for usage by low-SWaP systems. Simplifying 
assumptions include knowledge of the object 3D model, a non-cluttered scene and only considering
the pose of a single object.


# Motivation
We are interested in this problem as we want to predict the pose of an object using low SWaP systems. Current solutions to this problem require extensive computation and would not be feasible in a low SWaP system. There are several applications in which this approach would be beneficial, including quick obstacle avoidance and fast object trajectory estimation. 

Low SWaP systems include quadcopters where there is a restriction on computing power due to their small size. An example where our solution could be used would be a quadcopter trying to fly through a narrow gap in an object as shown in the figure below. In this situation, there is a limitation on computational power as the quadcopter has to be light and the algorithm also needs to be fast due to high-speed flight.

![Quadcopter](./assets/narrowObjectQuadcopter.png)

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

![Cereal Box](./assets/cereal_box.png)

<video width="320" height="240" controls>
    <source src="https://drive.google.com/file/d/1SqLvUjl3oCNT1KXPzjU2-KxbH9OFS9G3/view?usp=sharing" type="video/mp4">
</video>

## ORB Feature Extraction and Matching
For featuer extraction, we decided to go with ORB(Oriented FAST and Rotated BRIEF) features. ORB features are extracted from each reference image and stored. These 2D features have known 3D counterparts. This process is done in under 9 milliseconds.
![Orb Features](./assets/box_with_orb.png)

<video width="320" height="240" controls>
    <source src="https://drive.google.com/file/d/1SqLvUjl3oCNT1KXPzjU2-KxbH9OFS9G3/view?usp=sharing" type="video/mp4">
</video>

Next, ORB feature detection is computed over the new image, and feature matching is done between this 
new image and a reference image. We decided to go with a brute force matching approach using the 
minimum Hamming distance as it was able to match the orb features in under 0.5 milliseconds. Since 
the features in the reference image have known 3D values, this information is used for pose 
estimation described in the next section.

![Brute Force Matching](./assets/feature_matching.png)

## Pose Estimation
For pose estimation, we used Efficient Perspective-n-Point(EPnP) as shown in Figure 3. The EPnP
approach takes in the features in the image that coordinate to the 3D points as a weighted sum of four
virtual control points. This is less computationally heavy as it allows us to reduces the problem 
to estimating the position of these control points in the camera reference frame. The solution is 
an accurate non-iterative O(n) solution to the PnP problem. The main motivation for EPnP was the 
computation time of around 3 milliseconds. EPnP is much faster than other PnP methods even for 
large n values.

![EPnP](./assets/EPnP.png)


## Evaluation
We evaluated the performace of our pose estimation algorithm using 4 critereas.
1. Average Distance of Model Points (ADD): 
Calculates the average pairwise distance between the 3D model transformed according to the ground truth pose and the estimated pose.
![ADD](./assets/ADD_equation.png)

2. Average Closest Point Distance (ADD-S):
Calculates the average distance from each 3D model point transformed according to the estimated pose to its closest neighbour on the target model according to the ground truth pose.
![ADD-S](./assets/ADD-S_Equation.png)

Given ADD or ADD-S, we can calculate the area under the accuracy-threshold curve (AUC).
One approach is better than another if it yields higher AUC.

3. Relative Error: 
Calculates the relative error of the estimated rotation and translation
![Relative Error](./assets/Relative_Error_Equation.png)

4. Computational Speed:
To evaluate the computational speed of an approach, we run the codes of that approach 10 times on our testing dataset using Macbook Air, and then record their average run time. One approach is better than another if it takes shorter time.


## Results
INCLUDE RESULTS





## Reflection / Future Plans
Overall this project was a really great learning experience for both OpenCV and OpenGL simulation. 
In the future, we plan to implement this for applications such as a quadcomputer trying to fly 
through a narrow gap in an object.


## References


