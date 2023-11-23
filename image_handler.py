import cv2
import mediapipe as mp

import DB
from core import *

DB.signMapper.connect()
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
hand_options = HandLandmarkerOptions(
    num_hands=2,
    base_options=BaseOptions(model_asset_path='model/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.75,
    min_hand_presence_confidence=0.75)
HandLandmarker = mp.tasks.vision.HandLandmarker
hand_landmarker = HandLandmarker.create_from_options(hand_options)

PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/pose_landmarker.task'),
    output_segmentation_masks=False,
    running_mode=VisionRunningMode.IMAGE
)
PoseLandmarker = mp.tasks.vision.PoseLandmarker
pose_landmarker = PoseLandmarker.create_from_options(pose_options)

def extract_keyframes(image_path, prompt):
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    hand_results = hand_landmarker.detect(frame)
    pose_results = pose_landmarker.detect(frame)
    annotated_image = draw_landmarks_on_image(frame.numpy_view(), hand_results, pose_results)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    annotated_image = cv2.resize(annotated_image, (960, 680))
    cv2.imshow('MediaPipe Hands', annotated_image)
    normalized_landmarks = calculate(hand_results, pose_results, algorithm_a)
    print(normalized_landmarks)
    print(f"语义:{prompt}")
    # cv2.waitKey(1)
    # DB.signMapper.insert(normalized_landmarks, prompt)
    key = cv2.waitKey()
    if key == 32:
        DB.signMapper.insert(normalized_landmarks, prompt)
    else:
        with open("error.txt", mode="a", encoding="utf-8") as error:
            error.write(f"{prompt}\n")


with open("data/data.txt", encoding="utf-8") as file:
    datas = file.readlines()
    for data in datas:
        info = data.strip().split("|")
        extract_keyframes(info[1], info[0])
        print(f"处理:{info[0]}完成")
print("处理完成")
