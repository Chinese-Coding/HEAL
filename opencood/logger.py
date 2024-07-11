import sys

from loguru import logger

logger.remove()  # 删除默认的
logger.add(sys.stdout, level="SUCCESS")


def get_logger():
    return logger
