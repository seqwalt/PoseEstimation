#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_access.hpp>

#include <vector>

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

// Default camera values
const float YAW         = -90.0f;
const float PITCH       =  0.0f;
const float SPEED       =  2.5f;
const float SENSITIVITY =  0.1f;
const float ZOOM        =  45.0f; // fov degrees


// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
// Free Flying camera class. WASD keyboard control and change direction with mouse
class FreeCamera
{
public:
    // camera Attributes
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    // euler Angles
    float Yaw;
    float Pitch;
    // camera options
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;
    // Target
    glm::vec3 Target;

    // constructor with vectors for free-fly camera
    FreeCamera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
    {
        Position = position;
        WorldUp = up;
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }

    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 GetViewMatrix()
    {
        return glm::lookAt(Position, Position + Front, Up);
    }

    // processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void ProcessKeyboard(Camera_Movement direction, float deltaTime, bool FPS=false)
    {
        float velocity = MovementSpeed * deltaTime;
        if (FPS == false){
          if (direction == FORWARD)
              Position += Front * velocity;
          if (direction == BACKWARD)
              Position -= Front * velocity;
          if (direction == LEFT)
              Position -= Right * velocity;
          if (direction == RIGHT)
              Position += Right * velocity;
        }else{
          if (direction == FORWARD)
              Position += glm::normalize(glm::vec3(Front.x, 0.0, Front.z)) * velocity; // stay on xz plane and maintain velocity by normalization
          if (direction == BACKWARD)
              Position -= glm::normalize(glm::vec3(Front.x, 0.0, Front.z)) * velocity;
          if (direction == LEFT)
              Position -= Right * velocity;
          if (direction == RIGHT)
              Position += Right * velocity;
        }
    }

    // processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
    {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        Yaw   += xoffset;
        Pitch += yoffset;

        // make sure that when pitch is out of bounds, screen doesn't get flipped
        if (constrainPitch)
        {
            if (Pitch > 89.0f)
                Pitch = 89.0f;
            if (Pitch < -89.0f)
                Pitch = -89.0f;
        }

        // update Front, Right and Up Vectors using the updated Euler angles
        updateCameraVectors();
    }

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void ProcessMouseScroll(float yoffset)
    {
        Zoom += (float)yoffset;
        if (Zoom < 1.0f)
            Zoom = 1.0f;
        if (Zoom > 45.0f)
            Zoom = 45.0f;
    }

private:
    // calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors()
    {
        // calculate the new Front vector
        glm::vec3 front;
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        Front = glm::normalize(front);
        // also re-calculate the Right and Up vector
        Right = glm::normalize(glm::cross(Front, WorldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        Up    = glm::normalize(glm::cross(Right, Front));
    }
};

// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
// Target-locked camera class (camera always points toward target). Keys W/S to zoom in/out. Mouse to change view angle.
class TargetCamera
{
public:
    // camera Attributes
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    // euler Angles
    float Yaw;
    float Pitch;
    // camera options
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;

    // constructor with vectors for fixed-target camera
    TargetCamera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f)) : MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
    {
        Position = position;
        WorldUp = up;
        Target = target;
        Radius = glm::length(position - target);
        updateCameraVectors();
    }

    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 GetViewMatrix()
    {
        return glm::lookAt(Position, Target, WorldUp);
    }

    // processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void ProcessKeyboard(Camera_Movement direction, float deltaTime, bool FPS=false)
    {
        float radiusDelta = 2 * MovementSpeed * deltaTime;
        if (direction == FORWARD){
            if (Radius > radiusDelta){ // do not move through target
              Position -= Front * radiusDelta;
              Radius = glm::length(Position - Target);
            }
        }
        if (direction == BACKWARD){
            Position += Front * radiusDelta;
            Radius = glm::length(Position - Target);
        }
    }

    // processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
    {
        xoffset *= 3 * MouseSensitivity;
        yoffset *= 3 * MouseSensitivity;

        if (firstMouse){
          // calculate initial Yaw and Pitch values
          glm::vec3 radial = Position - Target;
          Yaw = glm::degrees(glm::atan(radial.z/radial.x)) - 180.f * (float)(radial.x < 0.0f);
          Pitch = glm::degrees(glm::asin(radial.y/Radius));
          firstMouse = false;
        }

        Yaw   += xoffset;
        Pitch -= yoffset;

        // make sure that when pitch is out of bounds, screen doesn't get flipped
        if (constrainPitch)
        {
            if (Pitch > 89.0f){
                Pitch = 89.0f;
            }
            if (Pitch < -89.0f){
                Pitch = -89.0f;
            }
        }

        // move on sphere of radius Radius around target
        Position.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch)) * Radius + Target.x;
        Position.y = sin(glm::radians(Pitch)) * Radius + Target.y;
        Position.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch)) * Radius + Target.z;

        // update Front, Right and Up Vectors
        updateCameraVectors();
    }

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void ProcessMouseScroll(float yoffset)
    {
        Zoom += (float)yoffset;
        if (Zoom < 1.0f)
            Zoom = 1.0f;
        if (Zoom > 45.0f)
            Zoom = 45.0f;
    }

    // set the target point for the camera to look at
    void setTarget(glm::vec3 target)
    {
      Target = target;
      // make cooresponding updates to Radius and camera vectors
      Radius = glm::length(Position - Target);
      updateCameraVectors();
    }

    float getYaw()
    {
      return Yaw;
    }

    float getPitch()
    {
      return Pitch;
    }

    glm::vec3 getPosition()
    {
      return Position;
    }

private:
    bool firstMouse = true; // first time using mouse
    float Radius; // Radius of camera movement
    glm::vec3 Target; // Target

    // calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors()
    {
        Front = glm::normalize(Position - Target);
        Right = glm::normalize(glm::cross(Front, WorldUp));
        Up    = glm::normalize(glm::cross(Right, Front));
    }
};

#endif
