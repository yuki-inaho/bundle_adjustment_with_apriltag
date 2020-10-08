import numpy as np
import cv2
from pathlib import Path
from utils import set_camera_parameter, set_ir_camera_parameter, colorize_depth_img
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
    parser.add_argument('--color-input-dir', '-ic', type=str, default=f'{PARENT_DIR}/data', \
                        help='location of captured color images')
    parser.add_argument('--depth-input-dir', '-id', type=str, default=f'{PARENT_DIR}/depth', \
                        help='location of captured depth images')
    parser.add_argument('--output-dir', '-o', type=str, default=f'{PARENT_DIR}/projected', \
                        help='location to save projection images')
    args = parser.parse_args()
    return args


def read_images(input_dir, frame_name_ary, is_uc16=False):
    images = []
    for frm_name in frame_name_ary:
        image_name = str(Path(input_dir, frm_name.replace(".csv", ".png")).resolve())
        if is_uc16:
            images.append(cv2.imread(image_name, cv2.IMREAD_ANYDEPTH))
        else:
            images.append(cv2.imread(image_name))
    return images


def draw_points(image, points, color, radius=10):
    for i in range(points.shape[0]):
        image = cv2.circle(image, (points[i, 0], points[i, 1]), radius, Color[color], -1)
    return image


# http://staff.www.ltu.se/~jove/courses/c0002m/least_squares.pdf
def calculate_plane_coefficients(points):
    pts_mean = points.mean(0)
    points_centorized = points - pts_mean
    S, sigma, Vt = np.linalg.svd(points_centorized)
    plane_coeff = Vt.T[:,-1]
    return plane_coeff, pts_mean


def projection(color_image, marker_points, camera_pose, camera_pose_pre, camera_param):
    K = camera_param.intrinsic_matrix
    projected_image = color_image.copy()
    camera_pose = camera_pose_pre @ camera_pose
    points_extd = np.c_[marker_points, np.repeat(1, marker_points.shape[0])]
    points_tfm = points_extd @ camera_pose.T[:, :3]
    #points_tfm[:,[0,1]] *= 1.2075
    points_tfm *= 0.039/0.03866119

    plane_coefficients, plane_origin = calculate_plane_coefficients(points_tfm)

    pts_3d_prj = points_tfm @ K.T
    x_pos = np.int16(pts_3d_prj[:,0]/pts_3d_prj[:,2])
    y_pos = np.int16(pts_3d_prj[:,1]/pts_3d_prj[:,2])
    pts_2d = np.c_[x_pos, y_pos]
    projected_image = draw_points(projected_image, pts_2d, color="orange", radius=7)
    hull = cv2.convexHull(pts_2d.reshape(-1,1,2).astype(np.int32))
    mask = np.zeros(color_image.shape, dtype=np.uint8)
    mask = cv2.cvtColor(cv2.drawContours(mask,[hull], 0, (255, 255, 255), -1), cv2.COLOR_RGB2GRAY)
    y_idx, x_idx = np.where(mask > 0)
    fx, fy = camera_param.focal
    cx, cy = camera_param.center
    u_idx = (x_idx - cx).astype(np.float)/fx
    v_idx = (y_idx - cy).astype(np.float)/fy

    uve = np.c_[u_idx, v_idx, np.repeat(1, u_idx.shape[0])]
    z_var = (plane_coefficients @ plane_origin)/(uve @ plane_coefficients) + 0.032
    # residual = (np.c_[u_idx*z_var, v_idx*z_var, z_var]  - plane_origin) @ plane_coefficients

    z_var_int16 = (z_var * 1000).astype(np.int16)
    plane_depth = np.zeros([color_image.shape[0], color_image.shape[1]], dtype=np.int16)
    plane_depth[y_idx, x_idx] = z_var_int16

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    ret2,th_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plane_depth[th_image==0] = 0
    plane_depth_colorized = colorize_depth_img(plane_depth)
    return plane_depth, plane_depth_colorized


def main(cfg_file_path, color_input_dir, depth_input_dir, output_dir):
    toml_dict = toml.load(open(cfg_file_path))
    camera_param = set_ir_camera_parameter(toml_dict)

    camera_pose_ary = np.loadtxt(f"{PARENT_DIR}/camera_pose_ary.csv")
    marker_points = np.loadtxt(f"{PARENT_DIR}/markers.csv")
    frame_name_ary = np.loadtxt(f"{PARENT_DIR}/frame_name_list.csv", dtype = "unicode")
    color_images = read_images(color_input_dir, frame_name_ary)
    depth_images = read_images(depth_input_dir, frame_name_ary, is_uc16=True)

    camera_pose_pre = np.eye(4)
    n_images = len(color_images)
    diffs = []
    for i, image in enumerate(color_images):
        print(f"{i}/{n_images}")
        camera_pose = camera_pose_ary[i, :].reshape(4,4)
        plane_depth, plane_depth_colorized = projection(
            image, marker_points, camera_pose, camera_pose_pre, camera_param
        )
        cv2.imwrite(f"{PARENT_DIR}/test.png", plane_depth_colorized)
        depth_image = depth_images[i]
        depth_image[plane_depth == 0] = 0
        diff = depth_image.astype(float) - plane_depth.astype(float)
        diff_abs = np.abs(diff).astype(np.int16)
        diff_colorized = colorize_depth_img(diff_abs)
        diff_ext = diff[diff>0]
        depth_image[plane_depth==0] = 0
        cv2.imwrite(f"{PARENT_DIR}/diff/{i}.png", diff_colorized)
        cv2.imwrite(f"{PARENT_DIR}/board/{i}.png", colorize_depth_img(plane_depth))
        cv2.imwrite(f"{PARENT_DIR}/raw/{i}.png", colorize_depth_img(depth_image))
        cv2.waitKey(10)

        #diffs.extend(diff_ext)
    #hist = np.histogram(diffs)

if __name__ == '__main__':
    args = parse_args()
    main(args.cfg_file, args.color_input_dir, args.depth_input_dir, args.output_dir)
