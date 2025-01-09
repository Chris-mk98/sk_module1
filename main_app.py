import mediapipe as mp
import numpy as np
import cv2
import os
import csv
import time
import datetime
from config import *

class DetectionLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_log_file = None
        self.has_detections = False

    def init_log_file(self, timestamp=None):
        """로그 파일 초기화 (실제 파일은 첫 탐지 시 생성)"""
        if timestamp:  # 웹캠 모드: avi 파일과 동일한 타임스탬프 사용
            self.current_log_file = os.path.join(self.log_dir, f"recording_{timestamp}.csv")
        else:  # 비디오 모드: 새로운 타임스탬프 생성
            self.current_log_file = os.path.join(self.log_dir, f"detection_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.has_detections = False

    def log_detection(self, category_name: str, timestamp, confidence: float):
        """탐지 결과 로깅"""
        if not self.has_detections:
            # 첫 탐지 시에만 파일 생성 및 헤더 작성
            with open(self.current_log_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Category Name", "Timestamp", "Confidence"])
            self.has_detections = True
        
        # 탐지 결과 추가
        with open(self.current_log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([category_name, timestamp, confidence])

    def get_current_log_file(self):
        """현재 로그 파일 경로 반환"""
        return self.current_log_file if self.has_detections else None

class ObjectDetectorApp:
    def __init__(self):
        self.setup_directories()
        self.setup_detector()
        self.logger = DetectionLogger(REALTIME_LOG_DIR if INPUT_MODE == 0 else VIDEO_LOG_DIR)
        self.setup_video_capture()
        self.video_writer = None
        self.is_first_frame = True
        if INPUT_MODE != 0:
            self.logger.init_log_file()

    def setup_directories(self):
        for directory in [LOG_DIR, VIDEO_LOG_DIR, REALTIME_LOG_DIR]:
            os.makedirs(directory, exist_ok=True)

    def setup_detector(self):
        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=base_options,
            max_results=5,
            running_mode=mp.tasks.vision.RunningMode.IMAGE
        )
        self.detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

        second_base_options = mp.tasks.BaseOptions(model_asset_path=MODEL2_PATH)
        second_options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=second_base_options,
            max_results=5,
            running_mode=mp.tasks.vision.RunningMode.IMAGE
        )
        self.human_detector = mp.tasks.vision.ObjectDetector.create_from_options(second_options)
        
        # 이미지 분류기 초기화
        classifier_options = mp.tasks.vision.ImageClassifierOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=CLASSIFIER_MODEL_PATH)
        )
        self.classifier = mp.tasks.vision.ImageClassifier.create_from_options(classifier_options)

    def setup_video_capture(self):
        """비디오 캡처 초기화"""
        self.cap = cv2.VideoCapture(INPUT_MODE)
        if not self.cap.isOpened():
            raise ValueError("비디오를 열 수 없습니다. 경로를 확인하세요.")

        if INPUT_MODE == 0:
            # 웹캠 해상도 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # 첫 프레임을 기다리기 위해 여기서는 video_writer를 초기화하지 않음

    def setup_video_writer(self, first_frame):
        """비디오 작성기 초기화 및 새로운 파일 생성"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 비디오 파일 설정
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 타임스탬프를 공유하여 avi와 csv 파일명을 동일하게 생성
        self.timestamp = timestamp
        self.output_filename = os.path.join(REALTIME_LOG_DIR, f"recording_{timestamp}.avi")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_height, frame_width = first_frame.shape[:2]
            
        if hasattr(self, 'video_writer') and self.video_writer:
            self.video_writer.release()
            
        self.video_writer = cv2.VideoWriter(
            self.output_filename, 
            fourcc, 
            fps, 
            (frame_width, frame_height)
        )
        
        # 로그 파일 설정 (파일은 첫 탐지 시 생성됨)
        self.logger.init_log_file(timestamp if INPUT_MODE == 0 else None)
        self.start_time = time.time()

    def visualize(self, image, detection_result):
        for detection in detection_result:
            bbox = detection.bounding_box
            category = detection.categories[0]
            category_name = category.category_name.strip()
            name_to_korean = {'scrofa': 'boar', 'inermis': 'k_deer', 'thibetanus': 'bear', 'boar': 'boar',
                            'pygargus': 'k_deer', 'procyonoides': 'raccoon', 'sibirica': 'weasel'}
            # 영단어를 한국어로 변환
            display_name = name_to_korean.get(category_name.lower(), category_name)
            probability = round(category.score, 2)

            if probability < PROBABILITY_THRESHOLD:
                continue

            start_point = int(bbox.origin_x), int(bbox.origin_y)
            end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
            
            is_harmful = category_name.lower() in [animal.lower() for animal in HARMFUL_ANIMALS]
            color = TEXT_COLOR_RED if is_harmful else TEXT_COLOR_BLUE
            
            if is_harmful:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S") if INPUT_MODE == 0 else self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if self.logger.current_log_file is None:
                    self.logger.init_log_file()
                self.logger.log_detection(category_name, timestamp, probability)

            cv2.rectangle(image, start_point, end_point, color, 3)
            result_text = f"{display_name} ({probability})"
            print('Detected : ', result_text)
            cv2.putText(image, result_text, (start_point[0] + MARGIN, start_point[1] - ROW_SIZE), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, color, FONT_THICKNESS)

        if INPUT_MODE == 0:
            cv2.putText(image, time.strftime("%Y-%m-%d %H:%M:%S"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image
    
    def merge_detections(self, detections1, detections2):
        """두 모델의 탐지 결과를 병합하여 중복된 탐지를 제거하고 신뢰도 높은 결과만 선택"""
        merged_detections = []
        iou_threshold = 0.5  # 중복 탐지 여부를 결정할 IOU 임계값

        # 두 결과 리스트 병합 및 중복 제거
        for det1 in detections1:
            best_detection = det1
            for det2 in detections2:
                iou = self.calculate_iou(det1.bounding_box, det2.bounding_box)
                if iou > iou_threshold:
                    # 동일한 객체로 판단되면 신뢰도 높은 결과 선택
                    best_detection = det1 if det1.categories[0].score > det2.categories[0].score else det2
            merged_detections.append(best_detection)

        return merged_detections

    def calculate_iou(self, bbox1, bbox2):
        """두 바운딩 박스 간의 IOU(Intersection Over Union) 계산"""
        x1 = max(bbox1.origin_x, bbox2.origin_x)
        y1 = max(bbox1.origin_y, bbox2.origin_y)
        x2 = min(bbox1.origin_x + bbox1.width, bbox2.origin_x + bbox2.width)
        y2 = min(bbox1.origin_y + bbox1.height, bbox2.origin_y + bbox2.height)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0    
    

    def run(self):
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break

                # 첫 프레임에서 video_writer 초기화
                if INPUT_MODE == 0 and self.is_first_frame:
                    self.setup_video_writer(frame)
                    self.is_first_frame = False

                # 프레임 처리
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # 멧돼지 탐지 (detector)
                wild_boar_detections = [
                    detection for detection in self.detector.detect(mp_image).detections 
                    if detection.categories[0].category_name.lower() in [animal.lower() for animal in HARMFUL_ANIMALS]
                ]

                # 사람 탐지 (human_detector)
                person_detections = [
                    detection for detection in self.human_detector.detect(mp_image).detections 
                    if detection.categories[0].category_name.lower() == "person"
                ]

                # 동일 객체에 대해 더 높은 점수를 가진 탐지만 유지
                merged_detections = self.merge_detections(wild_boar_detections, person_detections)

                # 결과 시각화
                frame = self.visualize(frame, merged_detections)             

                # 실시간 녹화 처리
                if INPUT_MODE == 0:
                    current_time = time.time()
                    if current_time - self.start_time >= RECORD_PERIOD:
                        self.setup_video_writer(frame)  # 새 파일 생성
                    
                    if self.video_writer and self.video_writer.isOpened():
                        self.video_writer.write(frame)

                # 결과 표시
                cv2.namedWindow('MediaPipe Object Detection', cv2.WINDOW_NORMAL)
                cv2.imshow('MediaPipe Object Detection', frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        self.detector.close()

if __name__ == "__main__":
    app = ObjectDetectorApp()
    app.run()
