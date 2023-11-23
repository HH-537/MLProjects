import logging
import os
from datetime import datetime

LOG_FILE_NAME = f'{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.log'
log_file_path = os.path.join(os.getcwd(),"logs", LOG_FILE_NAME)
os.makedirs(log_file_path, exist_ok=True)

LOG_FP = os.path.join(log_file_path, LOG_FILE_NAME)
logging.basicConfig(
    filename=LOG_FP,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)