from camera_parameter import CameraParam

def set_camera_parameter(toml_dict):
    camera_param = CameraParam()
    intrinsics = [toml_dict["Camera0_Factory"][elem] for elem in ["fx", "fy", "cx", "cy"]]
    camera_param.set_intrinsic_parameter(*intrinsics)
    image_size = [toml_dict["Camera0"][elem] for elem in ["width", "height"]]
    camera_param.set_image_size(*image_size)
    return camera_param

