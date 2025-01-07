import mediapipe as mp
import numpy as np
import cv2

# MediaPipe 모델 경로
model_path = r"C:\Users\user\sk_module1\model\efficientdet_lite0.tflite"

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
TEXT_COLOR_BLUE = (255, 0, 0) # 텍스트 색상 (파랑색)
TEXT_COLOR_RED = (0, 0, 255)  # 텍스트 색상 (빨간색)

# 위험 동물 리스트
HARMFUL_ANIMALS= ['person', 'bear']


def visualize(image, detection_result, harmful_animals) -> np.ndarray:
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        category = detection.categories[0]
        category_name = category.category_name.strip()  # 공백 제거

        # 라벨과 점수 표시
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (start_point[0] + MARGIN, start_point[1] - ROW_SIZE)

        # 경계 상자 좌표 설정
        start_point = int(bbox.origin_x), int(bbox.origin_y)
        end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)

        # 위험 동물 여부 판단
        if category_name.lower() in [animal.lower() for animal in harmful_animals]:  
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR_RED, 3)
            result_text_color = TEXT_COLOR_RED
        else:
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR_BLUE, 3)
            result_text_color = TEXT_COLOR_BLUE
            
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, result_text_color, FONT_THICKNESS)
        print(f"Detected: {category_name} with score: {probability}")  # 디버깅용 출력

    return image

# 비디오 파일 로드
video_capture = cv2.VideoCapture(r'C:\Users\user\sk_module1\resource\video\test.mp4') # 입력 비디오 파일 경로
fps = video_capture.get(cv2.CAP_PROP_FPS)  # 비디오의 프레임 속도(FPS) 가져오기
if not video_capture.isOpened():
    print("비디오를 열 수 없습니다. 경로를 확인하세요.")  # 비디오가 열리지 않을 경우 메시지 출력
    exit()



# MediaPipe ObjectDetector 생성
with ObjectDetector.create_from_options(options) as detector:
    while video_capture.isOpened():
        # 다음 프레임 읽기
        success, frame = video_capture.read()
        if not success:
            print("비디오 끝 또는 읽기 실패.")  # 더 이상 읽을 프레임이 없을 경우 루프 종료
            break

        # OpenCV 프레임을 MediaPipe 이미지로 변환
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV의 BGR 이미지를 RGB로 변환
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)  # MediaPipe 이미지 생성

            # 객체 탐지 수행
            detection_result = detector.detect(mp_image)  # detect 메서드로 탐지 수행

            # 탐지 결과 시각화
            image_with_detections = visualize(frame, detection_result, HARMFUL_ANIMALS)

            # 창크기 조절
            cv2.namedWindow('MediaPipe Object Detection', flags=cv2.WINDOW_NORMAL)

            # 결과 출력
            cv2.imshow('MediaPipe Object Detection', image_with_detections)  # 탐지된 결과 출력

            # ESC 키로 종료
            if cv2.waitKey(5) & 0xFF == 27:
                break

# 자원 해제
video_capture.release()  # 비디오 캡처 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
