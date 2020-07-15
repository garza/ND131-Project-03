import os
import sys
import logging as log
import numpy as np
import os.path as ospath
import cv2
from src.frame_feed import FrameFeed
from src.mouse_controller import MouseController
from time import time
from argparse import ArgumentParser

log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-it", "--input_type", required=True, type=str, default="video",
                        help="video or cam for camera")
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-o", "--output_path", type=str, default="./output",
                        help="Output path for any generated content"
                        "(./output by default)")
    return parser

def infer_from_stream(args, stats):
    log.info("Infer on Stream!")

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    log.info("Main Start")
    log.info(args.input)
    inference_stats = []
    infer_from_stream(args, inference_stats)

    if args.input_type == 'cam':
        feeder = FrameFeed(input_type='cam')
    else:
        if not os.path.isfile(args.input):
            log.error("not able to locate input video:")
            log.error(args.input)
            exit(1)
        feeder = FrameFeed(input_type='video', input_file=args.input)

    feeder.load_data()
    # TODO: demo video is 30 FPS and 1920x1080, ideally we would want to query this info from cv2, maybe
    # add this functionality from our FrameFeed class?
    outputstream = cv2.VideoWriter(os.path.join('output.mp4'), cv2.VideoWriter_fourcc(*'avc1'), 30/10, (1920, 1080), True)
    total_frame_count = 0
    for ret, frame in feeder.next_frame():
        if not ret:
            break
        total_frame_count =+ 1
        BREAK_WAIT_KEY = cv2.waitKey(60)
        try:
            log.info("start inference tasks here")
        except Exception as err:
            log.error("encountered error")
            log.error(err)
            log.error("unable to complete inference tasks")
            continue

        input_image = cv2.resize(frame, (900,500))
        log.info("update preview here")
        cv2.imshow('POST inference', input_image)
        log.info("update output stream")
        outputstream.write(frame)

        if total_frame_count % 5 == 0:
            log.info("move mouse based on results here")

        if BREAK_WAIT_KEY == 27:
            break

        log.info("compute and store stats here")

    try:
        os.mkdir(args.output_path)
    except OSError as error:
        log.error("unable to create output path")
        log.error(error)

    log.info("output final inference stats")
    log.info(inference_stats)

    log.info("save inference stats as output file")

    log.info("finished processing input video file")
    cv2.destroyAllWindows()
    feeder.close()


if __name__ == '__main__':
    main()
