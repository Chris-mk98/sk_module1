import mediapipe as mp
import numpy as np
import cv2
from matplotlib import pyplot as plt

# 객체 탐지에 사용할 모델 파일 경로
model_path = 'efficientdet_lite0.tflite'

# 입력 이미지 파일 경로
IMAGE_FILE = 'dog222.png'

# 이미지 파일에서 입력 이미지를 불러옵니다.
mp_image = mp.Image.create_from_file(IMAGE_FILE)

# numpy 배열에서 입력 이미지를 불러오는 코드 (필요 시 사용 가능)
# mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

# MediaPipe 객체 탐지에서 사용할 옵션 정의
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 객체 탐지 옵션 설정
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),  # 모델 경로 설정
    max_results=5,  # 최대 탐지 결과 수 설정
    running_mode=VisionRunningMode.IMAGE  # 이미지를 기반으로 탐지 모드 설정
)

# 시각화를 위한 OpenCV 및 NumPy 설정
MARGIN = 10  # 텍스트와 경계 상자의 여백 (픽셀)
ROW_SIZE = 10  # 텍스트 줄 간격 (픽셀)
FONT_SIZE = 1  # 텍스트 크기
FONT_THICKNESS = 1  # 텍스트 두께
TEXT_COLOR = (255, 0, 0)  # 텍스트 및 상자 색상 (빨간색)

# 탐지 결과 시각화 함수
def visualize(
    image,
    detection_result
) -> np.ndarray:
    """
    입력 이미지에 탐지 결과(경계 상자 및 라벨)를 그려 반환합니다.
    
    Args:
        image: 입력 RGB 이미지.
        detection_result: 시각화할 탐지 결과 리스트.
    
    Returns:
        경계 상자가 그려진 이미지.
    """
    for detection in detection_result.detections:
        # 경계 상자 그리기
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y  # 상자의 시작점
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height  # 상자의 끝점
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)  # 경계 상자 그리기

        # 라벨 및 확률 그리기
        category = detection.categories[0]
        category_name = category.category_name  # 카테고리 이름
        probability = round(category.score, 2)  # 확률 (소수점 두 자리)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)  # 텍스트 위치
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)  # 텍스트 그리기

    return image

# 객체 탐지기를 옵션을 사용해 생성
with ObjectDetector.create_from_options(options) as detector:
    # 입력 이미지에서 객체 탐지 수행
    detection_result = detector.detect(mp_image)
    print(detection_result)  # 탐지 결과 출력
    
    # 탐지 결과를 시각화하기 위해 처리
    image_copy = np.copy(mp_image.numpy_view())  # 원본 이미지를 복사
    annotated_image = visualize(image_copy, detection_result)  # 시각화 적용
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # OpenCV 이미지를 RGB로 변환
    
    # Matplotlib을 사용해 시각화 결과 표시
    plt.imshow(rgb_annotated_image)
    plt.axis('off')  # 축 숨기기
    plt.show()
