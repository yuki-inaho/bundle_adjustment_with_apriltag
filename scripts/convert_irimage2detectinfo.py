
import argparse
from pathlib import Path
import toml

from april_detector import get_detector_module
from utils import set_camera_parameter, set_ir_camera_parameter
import cv2
import numpy as np
import pdb

SCRIPT_DIR = str(Path(__file__).resolve().parent)
PARENT_DIR = str(Path(SCRIPT_DIR).parent)

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cfg-file', '-c', type=str, default=f'{PARENT_DIR}/cfg/camera_parameter.toml', \
                        help='location of a camera parameter file')
    parser.add_argument('--input-dir', '-i', type=str, default=f'{PARENT_DIR}/data', \
                        help='location of captured color images')
    parser.add_argument('--output-dir', '-o', type=str, default=f'{PARENT_DIR}/detected', \
                        help='location to save apriltag detected results')
    parser.add_argument('--tag-size', '-s', type=float, default=0.034, \
                        help='the size of apriltag[m]')
    args = parser.parse_args()
    cfg_file_path = args.cfg_file
    input_dir = args.input_dir
    output_dir = args.output_dir
    tag_size = args.tag_size
    return cfg_file_path, input_dir, output_dir, tag_size

def read_and_detect(image_path_str, detector):
    image = cv2.imread(image_path_str)
    result = detector.run_detection(image)
    return result


def save_detection_result(detection_result, name, output_dir, camera_param):
    result_ary = np.c_[np.asarray(detection_result[0]), np.asarray(detection_result[1])]
    save_name = str(Path(output_dir, name))
    np.savetxt(save_name, result_ary)


def main(cfg_file_path, input_dir, output_dir, tag_size):
    toml_dict = toml.load(open(cfg_file_path))
    image_path_str_list = [str(pt) for pt in list(Path(input_dir).glob("*.png"))]
    camera_param = set_ir_camera_parameter(toml_dict)
    detector = get_detector_module(camera_param, tag_size, b"36h11")

    n_data = len(image_path_str_list)
    for i, image_path_str in enumerate(image_path_str_list):
        base_name = Path(image_path_str).name
        base_name = base_name.replace('.png', '.csv')
        print(f"{i}/{n_data}")
        detection_result = read_and_detect(image_path_str, detector)
        save_detection_result(detection_result, base_name, output_dir, camera_param)


if __name__ == "__main__":
    cfg_file_path, input_dir, output_dir, tag_size = parse_args()
    main(cfg_file_path, input_dir, output_dir, tag_size)