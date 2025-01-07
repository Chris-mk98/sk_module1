import os

# 기본 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, "log")
VIDEO_LOG_DIR = os.path.join(LOG_DIR, "recorded_video_log")
REALTIME_LOG_DIR = os.path.join(LOG_DIR, "video_backup")

# 모델 경로 설정
MODEL_PATH = r"C:\Users\user\sk_module1\model\efficientdet_lite0.tflite"

# 시각화 설정
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR_BLUE = (255, 0, 0)
TEXT_COLOR_RED = (0, 0, 255)

# 탐지 설정
HARMFUL_ANIMALS = ['person', 'bear']
PROBABILITY_THRESHOLD = 0.5
INPUT_MODE = 0 # r"C:\Users\user\sk_module1\resource\video\test.mp4"
RECORD_PERIOD = 3600  # 1시간 