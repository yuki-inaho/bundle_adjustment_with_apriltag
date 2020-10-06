from april_detector_pywrapper import PyAprilTagDetector
from scipy.spatial.transform import Rotation as Rot
import numpy as np
from camera_parameter import CameraParam

class TagInfo:
    def __init__(self, tagid, corners, center, translation, rot_dcm_flatten):
        self.tagid = int(tagid)
        self.corners = corners.reshape((4, 2))
        self.center = center
        self.translation = translation
        self.rot_dcm = rot_dcm_flatten.reshape((3, 3))[[1, 2, 0], :][:, [1, 2, 0]]


class TagDetector:
    def __init__(self, camera_param, tagSize, tagFamily):
        self._camera_param = camera_param
        self._K = self._camera_param.intrinsic_matrix
        self.detector = PyAprilTagDetector()
        self.detector.setImageSize(*self._camera_param.image_size)
        self.detector.setCameraParams(*self._camera_param.intrinsics)
        self.detector.setTagSize(tagSize)
        self.detector.setTagCodes(tagFamily)
        self.detector.setup()

    def run_detection(self, img):
        self.detector.getDetectedInfo(img, draw_flag=False)
        taginfo_list = self.detector.extractTagInfo()
        tagobj_list = [self.taginfo_convert(_taginfo) for _taginfo in taginfo_list]
        tag_id_list = [_tagobj.tagid for _tagobj in tagobj_list]
        tag_translation_list = [self.get_translation(_tagobj) for _tagobj in tagobj_list]
        return tag_id_list, tag_translation_list

    def get_translation(self, tagobj):
        ''' 3D position of detect marker according to  [x, y, z] = [horisontal, vertical, depth] right-hand coordinate
        '''
        tag_tx = -tagobj.translation[1]
        tag_ty = -tagobj.translation[2]
        tag_tz = tagobj.translation[0]
        return [tag_tx, tag_ty, tag_tz]

    def taginfo_convert(self, _taginfo):
        return TagInfo(_taginfo[0], # ID
                       _taginfo[1:9], # Rotation matrix
                       _taginfo[9:11], # translation
                       _taginfo[11:14],
                       _taginfo[14:])

    @property
    def K(self):
        return self._K


def get_detector_module(camera_param: CameraParam, tag_size, tag_family):
    return TagDetector(camera_param, tag_size, tag_family)

