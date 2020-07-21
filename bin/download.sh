#!/bin/sh

model_dir="./models/"

omz_dl="/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py"

face_detection_bin="face-detection-adas-binary-0001"
face_detection="face-detection-adas-0001"
head_position="head-pose-estimation-adas-0001"
land_marks="landmarks-regression-retail-0009"
gaze_estimation="gaze-estimation-adas-0002"

mkdir ${model_dir}

#download our models
python3 ${omz_dl} --name ${face_detection_bin} -o ${model_dir}
python3 ${omz_dl} --name ${face_detection} -o ${model_dir}
python3 ${omz_dl} --name ${head_position} -o ${model_dir}
python3 ${omz_dl} --name ${land_marks} -o ${model_dir}
python3 ${omz_dl} --name ${gaze_estimation} -o ${model_dir}

echo "open model zoo models downloaded to ${model_dir}"
echo "Downloads Complete"
