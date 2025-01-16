import os

# 기본 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, "log")
VIDEO_LOG_DIR = os.path.join(LOG_DIR, "recorded_video_log")
REALTIME_LOG_DIR = os.path.join(LOG_DIR, "realtime_video_log")

# 모델 경로 설정
# MODEL_PATH = r"C:\Users\user\sk_module1\model\fixed_animal_model.tflite"
MODEL_PATH = r"C:\Users\user\sk_module1\model\boar_model.tflite"

MODEL2_PATH = r"C:\Users\user\sk_module1\model\efficientdet_lite0.tflite"

CLASSIFIER_MODEL_PATH = r"C:\Users\user\sk_module1\model\2.tflite"

# 시각화 설정
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 2
FONT_THICKNESS = 3
TEXT_COLOR_BLUE = (255, 0, 0)
TEXT_COLOR_RED = (0, 0, 255)

# 탐지 설정
HARMFUL_ANIMALS = ['scrofa', 'coreanus', 'inermis', 'thibetanus', 'boar', 'pygargus', 'procyonoides', 'thibetanus', 'sibirica' ]
PROBABILITY_THRESHOLD = 0.5
RECORD_PERIOD = 10

# INPUT_MODE = r"C:\Users\user\sk_module1\resource\video\시연영상\Kdeer_people.mp4"
# INPUT_MODE =  r"C:\Users\user\sk_module1\resource\video\시연영상\test.mp4"
# INPUT_MODE = r"C:\Users\user\sk_module1\resource\video\시연영상\난이도(중)3.mp4"
INPUT_MODE = 0



