import os
import sys
import logging as log
import numpy as np
import os.path as ospath
import cv2
from src.frame_feed import FrameFeed
from src.face_detection import FaceDetector
from src.head_pose_estimation import PoseDetector
from src.facial_landmarks_detection import LandmarkDetector
from src.mouse_controller import MouseController
from time import time
from argparse import ArgumentParser

log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO)
RED_MSG_COLOR = (0, 0, 255)

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
    parser.add_argument("-mf", "--face_model", required=True, type=str,
                        help="Path to an xml file with a trained model for Face Detector.")
    parser.add_argument("-ml", "--landmark_model", required=True, type=str,
                        help="Path to an xml file with a trained model for Landmark Detector.")
    parser.add_argument("-mp", "--pose_model", required=True, type=str,
                        help="Path to an xml file with a trained model for Pose Detector.")
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

def load_models(args, paths):
    models = {}
    face_model = FaceDetector(args.face_model, args.device, args.prob_threshold)
    face_model.check_model()
    face_model.load_model()
    pose_model = PoseDetector(args.pose_model, args.device)
    pose_model.check_model()
    pose_model.load_model()
    landmark_model = LandmarkDetector(args.landmark_model, args.device)
    landmark_model.check_model()
    landmark_model.load_model()

    models["face"] = face_model
    models["pose"] = pose_model
    models["landmark"] = landmark_model
    return models

def draw_face(frame, face):
    cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), RED_MSG_COLOR, 3)

def draw_pose(frame, head_pose):
    yaw_msg = "y:{:.1f}".format(head_pose[0])
    pitch_msg = "p:{:.1f}".format(head_pose[1])
    roll_msg = "r:{:.1f}".format(head_pose[2])
    cv2.putText(frame, yaw_msg, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 1.0, RED_MSG_COLOR, 2)
    cv2.putText(frame, pitch_msg, (15, 80), cv2.FONT_HERSHEY_COMPLEX, 1.0, RED_MSG_COLOR, 2)
    cv2.putText(frame, roll_msg, (15, 115), cv2.FONT_HERSHEY_COMPLEX, 1.0, RED_MSG_COLOR, 2)

def draw_eyes(frame, eyes, face):
    ##Adjust eye positions to frame instead of cropped face image
    le = eyes["eye_left"]
    le[0] = le[0] + face[0]
    le[1] = le[1] + face[1]
    re = eyes["eye_right"]
    re[0] = re[0] + face[0]
    re[1] = re[1] + face[1]
    #delta for drawing box
    d = 7
    #draw left eye
    cv2.rectangle(frame, (le[0] - d, le[1] + d), (le[0] + d, le[1] - d), RED_MSG_COLOR, 3)
    #draw right eye
    cv2.rectangle(frame, (re[0] - d, re[1] + d), (re[0] + d, re[1] - d), RED_MSG_COLOR, 3)
    ##Draw Right Eye Box
    ##cv2.rectangle(frame, (eyes[1][0], eyes[1][1]), (eyes[1][2], eyes[1][3]), RED_MSG_COLOR, 3)

def main():
    """
    our main run loop, get frame from FrameFeed
    and process each one thru our inference models
    draw our results to our frame and output to screen/save file
    """
    # Grab command line args
    args = build_argparser().parse_args()
    model_paths = {}
    face_model = args.face_model
    model_paths["face"] = args.face_model
    # Perform inference on the input stream
    log.info("Main Start")
    log.info(args.input)
    inference_stats = []
    models = load_models(args, model_paths)

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
        inference_preview = frame.copy()
        BREAK_WAIT_KEY = cv2.waitKey(60)
        try:
            log.info("start inference tasks here")
            #start_inf_time = time.time()
            face, face_img = models["face"].predict(frame.copy())
            if face == 0:
                continue
            draw_face(inference_preview, face)
            head_pose = models["pose"].predict(face_img)
            draw_pose(inference_preview, head_pose)
            eyes = models["landmark"].predict(face_img)
            log.info("eyes returned")
            log.info(eyes)
            draw_eyes(inference_preview, eyes, face)

        except Exception as err:
            log.error("encountered error")
            log.error(err)
            log.error("unable to complete inference tasks")
            continue

        input_image = cv2.resize(inference_preview, (960,540))
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
