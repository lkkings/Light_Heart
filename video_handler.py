import cv2
import mediapipe as mp
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog

from sklearn.preprocessing import MinMaxScaler

import DB

DB.signMapper.connect()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

# 创建 MinMaxScaler 对象
scaler = MinMaxScaler(feature_range=(0, 1))

def extract_keyframes(video_path, output_folder):
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    # 读取视频帧
    success, frame = video_capture.read()
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    normalized_hand_landmarks = []
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[0:int(0.8*frame.shape[0])]
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            hand_landmark_arr = []
            for hand_landmarks in results.multi_hand_world_landmarks:
                for hand_landmark in hand_landmarks.landmark:
                    x, y, z = hand_landmark.x, hand_landmark.y, hand_landmark.z
                    hand_landmark_arr.append(x)
                    hand_landmark_arr.append(y)
                    hand_landmark_arr.append(z)
            cur_hand_landmark_arr = np.array(hand_landmark_arr)
            cur_hand_landmark_arr = scaler.fit_transform(cur_hand_landmark_arr.reshape(-1, 1)).flatten()
            normalized_hand_landmarks = np.pad(cur_hand_landmark_arr, (0, 42 * 3 - len(hand_landmark_arr)),
                                               mode='constant',
                                               constant_values=float(-1))
            for hand_landmarks in results.multi_hand_landmarks:
                # 关键点可视化
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('MediaPipe Hands', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            def show_input_box(arr):
                root = tk.Tk()
                root.withdraw()  # Hide the main window

                result = simpledialog.askstring("语义", "请输入语义:")
                if not result:
                    return
                DB.signMapper.insert(normalized_hand_landmarks,result)
                print("You entered:", result)
                print("normalized_hand_landmarks:", arr)
                root.destroy()
            show_input_box(normalized_hand_landmarks)
        if key == 97:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, video_capture.get(cv2.CAP_PROP_POS_FRAMES) - 30)
            print("后退30")
        if key == 100:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, video_capture.get(cv2.CAP_PROP_POS_FRAMES) + 30)
            print("前进30")

        # 读取下一帧
        success, frame = video_capture.read()

    # 释放资源
    video_capture.release()


# 视频文件路径
input_video_path = 'video/【教程】手语基础教程 (P1. 第01课.问候).flv'

# 输出关键帧的文件夹路径
output_keyframes_folder = r'D:\Project\Python\LightHeart\video\1'

# 提取关键帧
extract_keyframes(input_video_path, output_keyframes_folder)
