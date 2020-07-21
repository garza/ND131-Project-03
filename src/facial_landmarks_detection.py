'''
Implemented from model.py example class skeleton given as part of the project
'''
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork,IECore
import warnings

log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO)

class LandmarkDetector:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name='models/intel/facial-landmarks-35-adas-0002/FP32-INT8/facial-landmarks-35-adas-0002', device='CPU', extensions=None):
        self.model_name = model_name
        self.model_w = model_name + '.bin'
        self.model_s = model_name + '.xml'
        self.device = device
        self.extension = extensions
        self.inference_results = None
        self.pre_image = None
        self.post_image = None
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

    def predict(self, image):
        self.pre_image = self.preprocess_input(image)
        self.inference_results = self.exec_network.infer({self.input_name: self.pre_image})
        landmarks = self.preprocess_output(self.inference_results, image)
        return landmarks

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

    def preprocess_input(self, image):
        post_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        post_frame = post_frame.transpose((2, 0, 1))
        post_frame = post_frame.reshape(1, *post_frame.shape)
        input_width = post_frame.shape[1]
        input_height = post_frame.shape[0]
        log.debug("input width x height: %d x %d", self.input_shape[3], self.input_shape[2])
        return post_frame

    def preprocess_output(self, outputs, image):
        '''
        We only care about the eye coordinates for this project, parse the eye's (X,Y) positions
        and repackage for further processing
        '''
        width = image.shape[1]
        height = image.shape[0]
        outputs = outputs[self.output_name][0]
        ##Left Eye Position (X,Y)
        eye_l_x = int(outputs[0] * width)
        eye_l_y = int(outputs[1] * height)
        ##Right Eye Position (X,Y)
        eye_r_x = int(outputs[2] * width)
        eye_r_y = int(outputs[3] * height)
        nose_x = int(outputs[4] * width)
        nose_y = int(outputs[5] * height)
        lip_l_x = int(outputs[6] * width)
        lip_l_y = int(outputs[7] * height)
        lip_r_x = int(outputs[8] * width)
        lip_r_y = int(outputs[9] * height)
        ##used for later drawing
        result = {'eye_left': [eye_l_x, eye_l_y],
                  'eye_right': [eye_r_x, eye_r_y],
                  'nose': [nose_x, nose_y],
                  'lip_left': [lip_l_x, lip_l_y],
                  'lip_right': [lip_r_x, lip_r_y]
                  }
        return result
