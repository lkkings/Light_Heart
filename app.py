import asyncio
import signal
import mediapipe as mp

import DB
import Vits

from core import *

hand_landmarks_queue: asyncio.Queue
text_queue: asyncio.Queue
down_event: asyncio.Event

change_threshold = 0
confidence_threshold = 20
frame_rate = 3

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
hand_options = HandLandmarkerOptions(
    num_hands=2,
    base_options=BaseOptions(model_asset_path='model/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    min_hand_detection_confidence=0.75,
    min_hand_presence_confidence=0.75)
HandLandmarker = mp.tasks.vision.HandLandmarker
hand_landmarker = HandLandmarker.create_from_options(hand_options)

PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/pose_landmarker.task'),
    output_segmentation_masks=False,
    running_mode=VisionRunningMode.VIDEO
)
PoseLandmarker = mp.tasks.vision.PoseLandmarker
pose_landmarker = PoseLandmarker.create_from_options(pose_options)


async def detect():
    print("***************detect running***************")
    cap = cv2.VideoCapture(0)
    timestamp_ms = 0
    pre_landmarks = np.zeros(1000)
    while not down_event.is_set():
        timestamp_ms += 1
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame,1)
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        hand_results = hand_landmarker.detect_for_video(frame, timestamp_ms)
        pose_results = pose_landmarker.detect_for_video(frame, timestamp_ms)
        annotated_image = draw_landmarks_on_image(frame.numpy_view(), hand_results, pose_results)
        cv2.imshow('MediaPipe Hands', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == 27:
            pass
        if not hand_results.hand_landmarks:
            await text_queue.put("ok")
            continue
        if timestamp_ms % frame_rate != 0:
            continue
        normalized_landmarks = calculate(hand_results, pose_results, algorithm_a)
        # 通过欧式距离推测变化弧度
        distance = np.linalg.norm(normalized_landmarks - pre_landmarks[:len(normalized_landmarks)])
        if distance > change_threshold:
            await hand_landmarks_queue.put(normalized_landmarks)
        pre_landmarks = normalized_landmarks


async def identify():
    print("***************identify running***************")
    while not down_event.is_set():
        hand_landmarks = await hand_landmarks_queue.get()
        for i in DB.signMapper.search_by_id(hand_landmarks):
            print("distance:", i[0].distance)
            if i[0].distance > confidence_threshold:
                continue
            print("value:", i[0].fields["prompt"])
            await text_queue.put(i[0].fields["prompt"])


async def vits():
    print("***************vits running***************")
    pre_text = ""
    sentence = ""
    i = 0
    while not down_event.is_set():
        text = await text_queue.get()
        if pre_text == text or i < 3:
            i += 1
            continue
        if text != pre_text and text != "ok":
            i = 0
            sentence += text
            pre_text = text
        if text == "ok" and len(sentence) > 0:
            Vits.ttsbot.speak(sentence)
            print(sentence)
            sentence = ""
            i = 0


async def main():
    global down_event, hand_landmarks_queue, text_queue
    down_event = asyncio.Event()
    hand_landmarks_queue = asyncio.Queue(maxsize=1)
    text_queue = asyncio.Queue(maxsize=1)

    DB.signMapper.connect()

    # 设置中断信号
    def handle_sigint():
        down_event.set()


    bound_handle_sigint = lambda signum, frame: handle_sigint()
    signal.signal(signal.SIGINT, bound_handle_sigint)
    task3 = asyncio.create_task(vits())
    task1 = asyncio.create_task(identify())
    task2 = asyncio.create_task(detect())
    await asyncio.gather(task1, task2, task3)


if __name__ == '__main__':
    asyncio.run(main())
