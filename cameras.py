import numpy as np
import pybullet as p


def kinova_pusher_camera(camera):
    camera.CameraViewMatrix = p.computeViewMatrixFromYawPitchRoll([0.7,0.1,-0.00], 1, 90, -45, 0, 2)
    camera.CameraProjMatrix = p.computeProjectionMatrix(-0.5000, 0.5000, -0.5000, 1.5000, 1.0000, 6.0000)

