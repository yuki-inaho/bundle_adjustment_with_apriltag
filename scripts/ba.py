import toml
import g2o
import numpy as np
from collections import defaultdict
from utils import set_camera_parameter
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
    parser.add_argument('--output-dir', '-o', type=str, default=f'{PARENT_DIR}/detected', \
                        help='location to save apriltag detected results')
    args = parser.parse_args()
    return args


def initialize_camera_pose(optimizer, num_pose):
    poses = []
    for i in range(num_pose):
        # pose here means transform points from world coordinates to camera coordinates
        pose = g2o.SE3Quat(np.identity(3), [i*0.04-1, 0, 0])
        poses.append(pose)

        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(i)
        v_se3.set_estimate(pose)
        if i < 2:
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


def generate_detection_result_list(detection_result_path_str_list):
    marker_list_each_frame = []
    for detection_result_path_str in detection_result_path_str_list:
        result_ary = np.loadtxt(detection_result_path_str)
        if len(result_ary) == 0:
            continue
        id_list = result_ary[:, 0].astype(np.int16)
        translation_array = result_ary[:, 1:]
        marker_list = [DetectedMarker(tid, translation_array[i,:]) for i, tid in enumerate(id_list)]
        marker_list_each_frame.append(marker_list)
    return marker_list_each_frame

def main(use_robust_kernel, use_dense, cfg_file_path, output_dir):
    toml_dict = toml.load(open(cfg_file_path))
    camera_param = set_camera_parameter(toml_dict)

    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    focal_length = (camera_param.fx + camera_param.fy)/2
    principal_point = (camera_param.cx, camera_param.cy)
    cam = g2o.CameraParameters(focal_length, principal_point, 0)
    cam.set_id(0)
    optimizer.add_parameter(cam)

    detection_result_path_str_list = [str(pt) for pt in list(Path(output_dir).glob("*.csv"))]
    marker_list_each_frame = generate_detection_result_list(detection_result_path_str_list)

    num_pose = len(marker_list_each_frame)
    poses = initialize_camera_pose(optimizer, num_pose)
    point_id = num_pose
    inliers = dict()
    sse = defaultdict(float)

    for i, marker_list in enumerate(marker_list_each_frame):
        visible = []
        for j, marker in enumerate(poses):
            z = cam.cam_map(pose * marker.translation)
            if 0 <= z[0] < 640 and 0 <= z[1] < 480:
                visible.append((j, z))
        if len(visible) < 2:
            continue

        vp = g2o.VertexSBAPointXYZ()
        vp.set_id(point_id)
        vp.set_marginalized(True)
        vp.set_estimate(point + np.random.randn(3))
        optimizer.add_vertex(vp)

        inlier = True
        for j, z in visible:
            if np.random.random() < args.outlier_ratio:
                inlier = False
                z = np.random.random(2) * [640, 480]
            z += np.random.randn(2) * args.pixel_noise

            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, vp)
            edge.set_vertex(1, optimizer.vertex(j))
            edge.set_measurement(z)
            edge.set_information(np.identity(2))
            if args.robust_kernel:
                edge.set_robust_kernel(g2o.RobustKernelHuber())

            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)

        if inlier:
            inliers[point_id] = i
            error = vp.estimate() - true_points[i]
            sse[0] += np.sum(error**2)
        point_id += 1
    '''

    print('num vertices:', len(optimizer.vertices()))
    print('num edges:', len(optimizer.edges()))

    print('Performing full BA:')
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(10)


    for i in inliers:
        vp = optimizer.vertex(i)
        error = vp.estimate() - true_points[inliers[i]]
        sse[1] += np.sum(error**2)

    print('\nRMSE (inliers only):')
    print('before optimization:', np.sqrt(sse[0] / len(inliers)))
    print('after  optimization:', np.sqrt(sse[1] / len(inliers)))
    '''


if __name__ == '__main__':
    args = parse_args()
    main(args.robust, args.dense, args.cfg_file, args.output_dir)