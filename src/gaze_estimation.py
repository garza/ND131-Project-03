'''
Implemented from model.py example class skeleton given as part of the project
'''
import cv2
import numpy as np
import logging as log
import math
from openvino.inference_engine import IENetwork,IECore
import warnings

log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO)
EYE_INPUT_SIZE = (60, 60)

class GazeDetector:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name='models/intel/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002', device='CPU', extensions=None):
        self.model_name = model_name
        self.model_w = model_name + '.bin'
        self.model_s = model_name + '.xml'
        self.device = device
        self.extension = extensions
        self.inference_results = None
        self.pre_image_left = None
        self.pre_image_right = None
        self.network = None
        self.input_name = None
        self.input_shape = None
        self.output_shape = None
        self.output_name = None
        self.plugin = None
        self.exec_network = None

    def load_model(self):
        self.plugin = IECore()
        log.info("Attempting to load network for model:")
        log.info(self.model_name)
        self.network = self.plugin.read_network(model=self.model_s, weights=self.model_w)
        self.exec_network = self.plugin.load_network(self.network, self.device)
        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network.outputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_shape = self.network.outputs[self.output_name].shape

    def predict(self, left_eye, right_eye, pose):
        self.pre_image_left, self.pre_image_right = self.preprocess_input(left_eye, right_eye)
        self.inference_results = self.exec_network.infer({
            'left_eye_image': self.pre_image_left,
            'right_eye_image': self.pre_image_right,
            'head_pose_angles': pose
        })
        position, vector = self.preprocess_output(self.inference_results, pose)
        return position, vector

    def check_model(self):
        self.plugin = IECore()
        log.info("Checking model layers for model:")
        log.info(self.model_name)
        self.network = self.plugin.read_network(model=self.model_s, weights=self.model_w)
        # double check supported network layers
        # code taken from Project-01 (ND131)
        if "CPU" in self.device:
            supported_layers = self.plugin.query_network(self.network, "CPU")
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                        format(self.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
        versions = self.plugin.get_versions(self.device)
        log.info("{}{}".format(" "*8, self.device))
        log.info("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[self.device].major, versions[self.device].minor))
        log.info("{}Build ........... {}".format(" "*8, versions[self.device].build_number))

    def preprocess_input(self, left, right):
        post_left = cv2.resize(left, (EYE_INPUT_SIZE))
        post_left = post_left.transpose((2, 0, 1))
        post_left = post_left.reshape(1, *post_left.shape)
        post_right = cv2.resize(right, (EYE_INPUT_SIZE))
        post_right = post_right.transpose((2, 0, 1))
        post_right = post_right.reshape(1, *post_right.shape)
        return post_left, post_right

    def preprocess_output(self, outputs, pose):
        yaw = pose[0]
        pitch = pose[1]
        roll = pose[2]
        outputs = outputs[self.output_name][0]
        sint = math.sin(roll * math.pi / 180)
        cost = math.cos(roll * math.pi /180)
        x = outputs[0] * cost + outputs[1] * sint
        y = outputs[1] * cost + outputs[0] * sint
        return (x, y), outputs
