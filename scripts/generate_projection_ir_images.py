import numpy as np
import cv2
from pathlib import Path
from utils import set_camera_parameter, set_ir_camera_parameter
import toml
import argparse
import pdb


SCRIPT_DIR = str(Path(__file__).resolve().parent)
PARENT_DIR = str(Path(SCRIPT_DIR).parent)

Color = {
    "red": (0, 0, 255),
    "orange": (0, 64, 255),
    "blue": (255, 0, 0),
    "light_blue": (64, 255, 0),
    "green": (0, 255, 0),
    "black": (0, 0, 0)
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-file', '-c', type=str, default=f'{PARENT_DIR}/cfg/camera_parameter.toml', \
                        help='location of a camera parameter file')
    parser.add_argument('--input-dir', '-i', type=str, default=f'{PARENT_DIR}/data', \
                        help='location of captured color images')
    parser.add_argument('--output-dir', '-o', type=str, default=f'{PARENT_DIR}/projected', \
                        help='location to save projection images')
    args = parser.parse_args()
    return args


def read_images(input_dir, frame_name_ary):
    images = []
    for frm_name in frame_name_ary:
        image_name = str(Path(input_dir, frm_name.replace(".csv", ".png")).resolve())
        images.append(cv2.imread(image_name))
    return images


def draw_points(image, points, color, radius=10):
    for i in range(points.shape[0]):
        image = cv2.circle(image, (points[i, 0], points[i, 1]), radius, Color[color], -1)
    return image


def projection(color_image, marker_points, camera_pose, camera_pose_pre, camera_param):
    K = camera_param.intrinsic_matrix
    projected_image = color_image.copy()
    camera_pose = camera_pose_pre @ camera_pose
    points_extd = np.c_[marker_points, np.repeat(1, marker_points.shape[0])]
    points_tfm = points_extd @ camera_pose.T[:, :3]
    pts_3d_prj = points_tfm @ K.T

    x_pos = np.int16(pts_3d_prj[:,0]/pts_3d_prj[:,2])
    y_pos = np.int16(pts_3d_prj[:,1]/pts_3d_prj[:,2])
    pts_2d = np.c_[x_pos, y_pos]
    projected_image = draw_points(projected_image, pts_2d, color="orange", radius=7)
    return projected_image


def main(cfg_file_path, input_dir, output_dir):
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter(f'{PARENT_DIR}/projected_images.mp4', fmt, 4, (1280, 960)) # ライター作成

    toml_dict = toml.load(open(cfg_file_path))

    camera_param = set_ir_camera_parameter(toml_dict)

    camera_pose_ary = np.loadtxt(f"{PARENT_DIR}/camera_pose_ary.csv")
    marker_points = np.loadtxt(f"{PARENT_DIR}/markers.csv")
    frame_name_ary = np.loadtxt(f"{PARENT_DIR}/frame_name_list.csv", dtype = "unicode")
    color_images = read_images(input_dir, frame_name_ary)

    camera_pose_pre = np.eye(4)
    n_images = len(color_images)
    for i, image in enumerate(color_images):
        print(f"{i}/{n_images}")
        camera_pose = camera_pose_ary[i, :].reshape(4,4)
        projected_image = projection(image, marker_points, camera_pose, camera_pose_pre, camera_param)
        frame_name = frame_name_ary[i]
        frame_name = frame_name.replace('.csv', '.png')
        cv2.imwrite(f"{output_dir}/{frame_name}", projected_image)
        projected_image_resized = cv2.resize(projected_image, (1280, 960))
        writer.write(projected_image_resized)
        #writer.write(projected_image)
        cv2.waitKey(10)
    writer.release()


if __name__ == '__main__':
    args = parse_args()
    main(args.cfg_file, args.input_dir, args.output_dir)
