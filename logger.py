import os
import datetime
import logging

def set_logger(log_dir, log_fn=''):
    # 로그 생성
    logger = logging.getLogger()

    # 로그의 출력 기준 설정
    logger.setLevel(logging.INFO)

    # log 출력 형식
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # log 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # log를 파일에 출력
    os.makedirs(log_dir, exist_ok=True)
    if log_fn == '':
        log_path = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S.log"))
    else:
        log_path = os.path.join(log_dir, log_fn)
        
    print('Setting logger... Log is saved to {}'.format(log_path))

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger