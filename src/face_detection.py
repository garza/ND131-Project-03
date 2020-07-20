'''
Implemented from model.py example class skeleton given as part of the project
'''
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork,IECore
import warnings

log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO)

class FaceDetector:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name='models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001', device='CPU', threshold=0.6, extensions=None):
        self.model_name = model_name
        self.model_w = model_name + '.bin'
        self.model_s = model_name + '.xml'
        self.device = device
        self.threshold = threshold
        self.extension = extensions
        self.face_cropped = None
        self.face_positions = None
        self.inference_results = None
        self.first_face = None
        self.pre_image = None
        self.post_image = None
        self.network = None
        self.input_name = None
        self.input_shape = None
        self.output_shape = None
        self.output_shape = None
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
        self.face_positions = self.preprocess_output(self.inference_results, image)

        if len(self.face_positions) == 0:
            log.info("No faces detected")
            return 0, 0

        self.first_face = self.face_positions[0]
        self.face_cropped = image[self.first_face[1]:self.first_face[3],self.first_face[0]:self.first_face[2]]
        return self.first_face, self.face_cropped

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
        width = image.shape[1]
        height = image.shape[0]
        faces = []
        output_base = outputs[self.output_name][0][0]
        for face in output_base:
            face_confidence = face[2]
            if face_confidence >= self.threshold:
                xmin = np.int(width * face[3])
                ymin = np.int(height * face[4])
                xmax = np.int(width * face[5])
                ymax = np.int(height * face[6])
                #xmax = int((face[5] * image.shape[1])/4)
                #ymax = int((face[6] * image.shape[0])/1.5)
                faces.append([xmin, ymin, xmax, ymax])
        return faces
