import toml
import g2o
import cv2
import numpy as np
from typing import List
from collections import defaultdict
from utils import set_ir_camera_parameter
from visualizer import PangolinVisualizer
from pathlib import Path
import argparse
import pdb


SCRIPT_DIR = str(Path(__file__).resolve().parent)
PARENT_DIR = str(Path(SCRIPT_DIR).parent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robust', '-r', action='store_true', help='use robust kernel')
    parser.add_argument('--dense', '-d', action='store_true', help='use dense solver')
    parser.add_argument('--cfg-file', '-c', type=str, default=f'{PARENT_DIR}/cfg/camera_parameter.toml', \
                        help='location of a camera parameter file')
    parser.add_argument('--input-dir', '-i', type=str, default=f'{PARENT_DIR}/data', \
                        help='location of captured color images')
    parser.add_argument('--output-dir', '-o', type=str, default=f'{PARENT_DIR}/detected', \
                        help='location to save apriltag detected results')
    args = parser.parse_args()
    return args


def get_visible_marker_id_list(marker_list_each_frame):
    marker_id_list = [marker.id  for marker_list in marker_list_each_frame for marker in marker_list]
    return np.unique(marker_id_list)


def initialize_camera_pose(optimizer, num_pose):
    poses = []
    for i in range(num_pose):
        # pose here means transform points from world coordinates to camera coordinates
        pose = g2o.SE3Quat(np.identity(3), [0, 0, 0])
        poses.append(pose)

        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(i)
        v_se3.set_estimate(pose)
        if i == 0:
            v_se3.set_fixed(True)
        optimizer.add_vertex(v_se3)
    return poses

class DetectedMarker:
    def __init__(self, id:int, translation:np.ndarray):
        self._id = id
        self._translation = translation

    @property
    def id(self):
        return self._id

    @property
    def translation(self):
        return self._translation


class DetectedMarkerEachFrame:
    def __init__(self, marker_list: List[DetectedMarker]):
        self._setting(marker_list)

    def _setting(self, marker_list):
        self._visible_frame = []
        self._translation_each_frame = []
        for marker in marker_list:
            self._visible_frame.append(marker.id)
            self._translation_each_frame.append(marker.translation)

    @property
    def visible_frame(self):
        return self._visible_frame

    @property
    def translation_each_frame(self):
        return self._translation_each_frame



def generate_detection_result_list(detection_result_path_str_list):
    marker_list_each_frame = []
    frame_name_list = []
    detection_result_path_str_list = np.sort(detection_result_path_str_list)

    for i, detection_result_path_str in enumerate(detection_result_path_str_list):
        result_ary = np.loadtxt(detection_result_path_str)
        if len(result_ary) <= 4:
            continue
        id_list = result_ary[:, 0].astype(np.int16)
        translation_array = result_ary[:, 1:]
        marker_list = [DetectedMarker(tid, translation_array[i,:]) for i, tid in enumerate(id_list)]
        marker_list_each_frame.append(marker_list)
        frame_name_list.append(str(Path(detection_result_path_str).name))
    return marker_list_each_frame, frame_name_list

class MarkerBucket:
    def __init__(self, marker_list_each_frame, visible_marker_id_list):
        self._visible_marker_id_list = np.sort(visible_marker_id_list)
        self._visible_frame_and_translation_per_marker = {}
        self._setting(marker_list_each_frame)

    def _setting(self, marker_list_each_frame):
        self._marker_bucket = [[] for _ in self._visible_marker_id_list]
        self._visible_frame_and_translation_per_marker = {}
        for marker_id in self._visible_marker_id_list:
            self._visible_frame_and_translation_per_marker[marker_id] = []

        for frame_id, marker_list in enumerate(marker_list_each_frame):
            for marker in marker_list:
                self._visible_frame_and_translation_per_marker[marker.id].append((frame_id, marker.translation))

    def get_frame_id_and_translation_by_id(self, marker_id):
        return self._visible_frame_and_translation_per_marker[marker_id]


def read_images(input_dir):
    image_path_str_list = [str(pt) for pt in list(Path(input_dir).glob("*.png"))]
    images = [cv2.imread(image_path_str) for image_path_str in image_path_str_list]
    return images


def main(use_robust_kernel, use_dense, cfg_file_path, input_dir, output_dir):
    toml_dict = toml.load(open(cfg_file_path))
    color_images = read_images(input_dir)
    camera_param = set_ir_camera_parameter(toml_dict)

    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    focal_length = (camera_param.fx + camera_param.fy)/2
    principal_point = (camera_param.cx, camera_param.cy)
    image_size = camera_param.image_size
    cam = g2o.CameraParameters(focal_length, principal_point, 0)
    cam.set_id(0)
    optimizer.add_parameter(cam)

    detection_result_path_str_list = [str(pt) for pt in list(Path(output_dir).glob("*.csv"))]
    marker_list_each_frame, frame_name_list = generate_detection_result_list(detection_result_path_str_list)
    visible_marker_id_list = get_visible_marker_id_list(marker_list_each_frame)

    n_visible_marker = len(visible_marker_id_list)
    marker_bucket = MarkerBucket(marker_list_each_frame, visible_marker_id_list)

    num_pose = len(marker_list_each_frame)
    poses = initialize_camera_pose(optimizer, num_pose)
    point_id = num_pose
    inliers = dict()
    sse = defaultdict(float)

    def get_mean_marker_position(frame_and_translation_tuple_list):
        marker_translation_list = [_tuple[1] for _tuple in frame_and_translation_tuple_list]
        return np.asarray(marker_translation_list).mean(0)

    K = camera_param.intrinsic_matrix

    point_id = num_pose
    for marker_id in visible_marker_id_list:
        frame_and_translation_tuple_list = marker_bucket.get_frame_id_and_translation_by_id(marker_id)
        marker_translation = get_mean_marker_position(frame_and_translation_tuple_list)

        vp = g2o.VertexSBAPointXYZ()
        vp.set_id(point_id)
        vp.set_marginalized(True)
        vp.set_estimate(marker_translation)
        optimizer.add_vertex(vp)

        for _tuple in frame_and_translation_tuple_list:
            frame_id = _tuple[0]
            translation = _tuple[1]
            pose = poses[frame_id]
            #point_2d = cam.cam_map(translation)
            point_2d_tmp = translation @ K.T
            point_2d = np.array([int(point_2d_tmp[0]/point_2d_tmp[2]), int(point_2d_tmp[1]/point_2d_tmp[2])])
            is_valid = 0 <= point_2d[0] < image_size[0]  and 0 <= point_2d[1] < image_size[1]
            if not is_valid:
                continue

            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, vp)
            edge.set_vertex(1, optimizer.vertex(frame_id))
            edge.set_measurement(point_2d)
            edge.set_information(np.identity(2))
            #if use_robust_kernel:
            edge.set_robust_kernel(g2o.RobustKernelHuber())

            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)
        point_id += 1

    print('num vertices:', len(optimizer.vertices()))
    print('num edges:', len(optimizer.edges()))

    print('Performing full BA:')
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(300)

    camera_points = []
    for i in range(num_pose):
        vp = optimizer.vertex(i)
        camera_points.append(vp.estimate().matrix())

    marker_points = []
    for i in range(n_visible_marker):
        vp = optimizer.vertex(num_pose + i)
        marker_points.append(vp.estimate())

    visualizer = PangolinVisualizer()
    visualizer.set_data_for_visualization(camera_points, marker_points)
    visualizer.draw()


    camera_pose_ary = np.asarray(camera_points).reshape(-1, 16)
    frame_name_ary = np.asarray(frame_name_list)[:,np.newaxis]
    np.savetxt(f"{PARENT_DIR}/camera_pose_ary.csv", camera_pose_ary)
    np.savetxt(f"{PARENT_DIR}/markers.csv", marker_points)
    np.savetxt(f"{PARENT_DIR}/frame_name_list.csv", frame_name_ary, fmt="%s")

    '''
    print('\nRMSE (inliers only):')
    print('before optimization:', np.sqrt(sse[0] / len(inliers)))
    print('after  optimization:', np.sqrt(sse[1] / len(inliers)))
    '''


if __name__ == '__main__':
    args = parse_args()
    main(args.robust, args.dense, args.cfg_file, args.input_dir, args.output_dir)