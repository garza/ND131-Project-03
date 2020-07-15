'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class BaseModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self):
        ### Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.input_blob_name = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        self.output_name = None
        self.output_info = None
        self.feed_dict = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        ### Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        ### init inference engine plugin
        self.plugin = IECore()
        ### Add any necessary extensions ###
        log.info("attempting to add CPU extension!")
        log.info(cpu_extension)
        if cpu_extension and "CPU" in device:
            plugin_dir = '/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64'
            self.plugin.add_extension(cpu_extension, device)
            # plugin = IEPlugin(device=‘CPU', plugin_dirs=plugin_dir)
            # cpu_extension_path = '/home/temp/intel/openvino/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so'

        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        self.network = IENetwork(model=model_xml, weights=model_bin)
        ### Check for supported layers ###
        if "CPU" in device:
            supported_layers = self.plugin.query_network(self.network, "CPU")
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(args.device, ', '.join(not_supported_layers)))
                log.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)
        versions = self.plugin.get_versions(device)
        log.info("{}{}".format(" " * 8, device))
        log.info(
            "{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[device].major, versions[device].minor))
        log.info("{}Build ........... {}".format(" " * 8, versions[device].build_number))

        ### Return the loaded inference plugin ###
        self.network = IENetwork(model=model_xml, weights=model_bin)
        self.exec_network = self.plugin.load_network(self.network, device)

        ## grab the input layer ##
        self.input_blob = self.network.inputs
        ##next(iter(self.network.inputs))
        img_info_input_blob = None
        self.feed_dict = {}
        for blob_name in self.network.inputs:
            log.info("blob_name: " + blob_name)
            if len(self.network.inputs[blob_name].shape) == 4:
                self.input_blob_name = blob_name
            elif len(self.network.inputs[blob_name].shape) == 2:
                img_info_input_blob = blob_name
            else:
                raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                                   .format(len(net.inputs[blob_name].shape), blob_name))

        self.output_blob = next(iter(self.network.outputs))

        if img_info_input_blob:
            log.info("set input info")
            n, c, h, w = self.get_input_shape()
            self.feed_dict[img_info_input_blob] = [h, w, 1]

        self.output_name, self.output_info = "", self.network.outputs[next(iter(self.network.outputs.keys()))]
        for output_key in self.network.outputs:
            if self.network.layers[output_key].type == "DetectionOutput":
                self.output_name, self.output_info = output_key, self.network.outputs[output_key]

        if self.output_name == "":
            log.error("Can't find a DetectionOutput layer in the topology")

        log.info("load model done")
        ### Note: You may need to update the function parameters. ###
        log.info('Preparing output blobs')
        output_name, output_info = "", self.network.outputs[next(iter(self.network.outputs.keys()))]
        for output_key in self.network.outputs:
            log.info("iteration: %s", output_key)
            if self.network.layers[output_key].type == "DetectionOutput":
                output_name, output_info = output_key, self.network.outputs[output_key]

        if output_name == "":
            log.error("Can't find a DetectionOutput layer in the topology")

        output_dims = output_info.shape
        if len(output_dims) != 4:
            log.error("Incorrect output dimensions for SSD model")
        max_proposal_count, object_size = output_dims[2], output_dims[3]
        log.info("Max Proposal Count is: %s", max_proposal_count)
        log.info("Output Object Size: %s, %s", output_dims[2], output_dims[3])

        if object_size != 7:
            log.error("Output item should have 7 as a last dimension")

        output_info.precision = "FP32"
        return

    def get_input_blob(self):
        ##return self.network.inputs[self.input_blob]
        ##input_blob = None
        ##return input_blob
        ##self.network.inputs[input_blob].shape
        return self.input_blob

    def get_input_blob_name(self):
        ##return self.network.inputs[self.input_blob]
        ##input_blob = None
        ##return input_blob
        ##self.network.inputs[input_blob].shape
        return self.input_blob_name

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.network.inputs[self.input_blob_name].shape

    def get_output_name(self):
        return self.output_name

    def get_output_info(self):
        return self.output_info

    def exec_net(self, image):
        ### Start an asynchronous request ###
        ### Return any necessary information ###
        ##log.info("Start Async Inference")
        self.input_blob[self.input_blob_name] = image
        self.feed_dict[self.input_blob_name] = image
        self.exec_network.start_async(request_id=0, inputs=self.feed_dict)
        return

    def wait(self):
        ### Wait for the request to be complete. ###
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        ### Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        res = self.exec_network.requests[0].outputs[self.output_blob]
        return res

    def predict(self, image):
        exec_net(image)
        res = get_output()
        return res

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError
