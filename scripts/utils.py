from camera_parameter import CameraParam
import numpy as np
import cv2

def set_camera_parameter(toml_dict):
    camera_param = CameraParam()
    intrinsics = [toml_dict["Camera0_RGB_Factory"][elem] for elem in ["fx", "fy", "cx", "cy"]]
    camera_param.set_intrinsic_parameter(*intrinsics)
    image_size = [toml_dict["Camera0"][elem] for elem in ["width_rgb", "height_rgb"]]
    camera_param.set_image_size(*image_size)
    return camera_param


def set_ir_camera_parameter(toml_dict):
    camera_param = CameraParam()
    intrinsics = [toml_dict["Camera0_Factory"][elem] for elem in ["fx", "fy", "cx", "cy"]]
    camera_param.set_intrinsic_parameter(*intrinsics)
    image_size = [toml_dict["Camera0"][elem] for elem in ["width", "height"]]
    camera_param.set_image_size(*image_size)
    return camera_param


def colorize_depth_img(img, max_var=1000):
    img_colorized = np.zeros([img.shape[0], img.shape[1], 3]).astype(np.uint8)
    img_colorized[:, :, 1] = 255
    img_colorized[:, :, 2] = 255
    img_hue = img.copy().astype(np.float32)
    img_hue[np.where(img_hue > max_var)] = 0
    zero_idx = np.where((img_hue > max_var) | (img_hue == 0))
    img_hue *= 255.0/max_var
    img_colorized[:, :, 0] = img_hue.astype(np.uint8)
    img_colorized = cv2.cvtColor(img_colorized, cv2.COLOR_HSV2RGB)
    img_colorized[zero_idx[0], zero_idx[1], :] = 0
    return img_colorized
