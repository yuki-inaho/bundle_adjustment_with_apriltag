import OpenGL.GL as gl
import pangolin
import numpy as np
import cv2


class PangolinVisualizer:
    def __init__(self):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self._scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(1280, 720, 900, 900, 960, 540, 0.2, 200),
            pangolin.ModelViewLookAt(-0.5, 0.5, -0.5, 0, 0, 0, pangolin.AxisDirection.AxisY)
        )
        handler = pangolin.Handler3D(self._scam)

        self._dcam = pangolin.CreateDisplay()
        self._dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -1280.0/720.0)
        self._dcam.SetHandler(handler)


    def set_data_for_visualization(self, camera_pose_list, points):
        n_pose = len(camera_pose_list)
        self._camera_pose_list = camera_pose_list
        self._trajectory = np.asarray([self._camera_pose_list[i][:3,3] for i in range(n_pose)])
        self._points = np.asarray(points)

    def set_color_images(self, color_images):
        self._color_images = color_images

    def draw(self):

        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self._dcam.Activate(self._scam)

            # Draw Point Cloud
            gl.glPointSize(1)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawPoints(self._points)

            # Draw Point Cloud
            colors = np.zeros((len(self._points), 3))
            norms = np.sqrt(self._points[:, 0]**2 + self._points[:, 1]**2 + self._points[:, 2]**2)
            colors[:, 1] = 0
            colors[:, 2] = 0
            colors[:, 0] = 1
            gl.glPointSize(5)
            pangolin.DrawPoints(self._points, colors)

            # Draw lines
            gl.glLineWidth(1)
            gl.glColor3f(0.5, 0.8, 0.0)
            pangolin.DrawLine(self._trajectory)   # consecutive

            # Draw camera
            cpose = np.eye(4)

            gl.glLineWidth(1)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.DrawCameras(self._camera_pose_list, w=0.03)

            pangolin.FinishFrame()


