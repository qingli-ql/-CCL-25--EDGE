import logging
import logging.handlers
import sys
from datetime import datetime


def getLogger(cfg):
    logger = logging.getLogger("mylogger")
    logger.setLevel(logging.DEBUG)  # 设置最低日志级别为DEBUG

    # 配置 TimedRotatingFileHandler 用于所有级别的日志
    rf_handler = logging.handlers.TimedRotatingFileHandler(
        f"{cfg['file_path']}/all.log",
        when="midnight",
        interval=1,
        backupCount=7,
    )
    rf_handler.setLevel(logging.DEBUG)  # 设置此处理器的日志级别为DEBUG
    rf_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    # 配置 FileHandler 用于记录错误日志
    f_handler = logging.FileHandler(f"{cfg['file_path']}/error.log")
    f_handler.setLevel(logging.ERROR)  # 只记录ERROR级别及以上的日志
    f_handler.setFormatter(
        logging.Formatter(
            "\n%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"
        )
    )

    # 配置 StreamHandler 用于控制台输出f"{cfg['file_path']}/
    console_handler = logging.StreamHandler(sys.stdout)  # 输出到stdout
    console_handler.setLevel(logging.DEBUG)  # 控制台输出所有级别的日志
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    # 添加所有处理程序到logger
    logger.addHandler(rf_handler)
    logger.addHandler(f_handler)
    logger.addHandler(console_handler)

    return logger
