import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

# MediaPipe 모델 경로
model_path = r'C:\Users\user\module1\reference\efficientdet_lite0.tflite'

# MediaPipe ObjectDetector 옵션 설정
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),  # 모델 경로 설정
    max_results=5,  # 최대 결과 개수
    running_mode=VisionRunningMode.IMAGE,  # 이미지 모드로 실행
)

# 시각화 함수 정의
MARGIN = 10  # 텍스트와 경계 상자의 여백
ROW_SIZE = 10  # 텍스트의 줄 간격
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # 텍스트 색상 (빨간색)

def visualize(image, detection_result) -> np.ndarray:
    """
    입력 이미지에 탐지 결과(경계 상자와 라벨)를 그려주는 함수.

    Args:
        image: 입력 RGB 이미지.
        detection_result: MediaPipe ObjectDetector의 탐지 결과.

    Returns:
        경계 상자와 라벨이 그려진 이미지.
    """
    for detection in detection_result.detections:
        # 경계 상자 그리기
        bbox = detection.bounding_box
        start_point = int(bbox.origin_x), int(bbox.origin_y)  # 시작 좌표
        end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)  # 끝 좌표
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)  # 경계 상자 그리기

        # 라벨과 점수 그리기
        category = detection.categories[0]
        category_name = category.category_name  # 객체 이름
        probability = round(category.score, 2)  # 탐지 점수
        result_text = f"{category_name} ({probability})"  # 표시할 텍스트
        text_location = (start_point[0] + MARGIN, start_point[1] - ROW_SIZE)  # 텍스트 위치
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        # 감지된 객체 정보를 콘솔에 출력
        print(f"Detected: {category_name} with score: {probability}")

    return image

# 웹캠으로부터 비디오 캡처
video_capture = cv2.VideoCapture(0)  # 웹캠 장치 사용
if not video_capture.isOpened():
    print("웹캠을 열 수 없습니다. 장치를 확인하세요.")  # 웹캠이 열리지 않을 경우 메시지 출력
    exit()

# MediaPipe ObjectDetector 생성
with ObjectDetector.create_from_options(options) as detector:
    while video_capture.isOpened():
        # 다음 프레임 읽기
        success, frame = video_capture.read()
        if not success:
            print("프레임 읽기 실패.")  # 프레임을 읽지 못한 경우 루프 종료
            break

        # OpenCV 프레임을 MediaPipe 이미지로 변환
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV의 BGR 이미지를 RGB로 변환
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)  # MediaPipe 이미지 생성

            # 객체 탐지 수행
            detection_result = detector.detect(mp_image)  # detect 메서드로 탐지 수행

            # 탐지 결과 시각화
            image_with_detections = visualize(frame, detection_result)

            # 결과 출력
            cv2.imshow('MediaPipe Object Detection - Webcam', image_with_detections)  # 탐지된 결과 출력

            # ESC 키로 종료
            if cv2.waitKey(5) & 0xFF == 27:
                break

# 자원 해제
video_capture.release()  # 비디오 캡처 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
