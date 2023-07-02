import logging
import sys
import os

LOG_FMT = "[%(asctime)s] [%(filename)s] [line:%(lineno)d] %(levelname)s: %(message)s"
LOG_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

class Logger:
    def __init__(
        self,
        log_file_name="details.log",
        log_level=logging.DEBUG,
        log_dir="../logs",
        only_file=False,
    ):
        log_dir = os.path.join(os.path.dirname(__file__), log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self._logger = logging.getLogger()
        if not self._logger.handlers:
            self.formatter = logging.Formatter(fmt=LOG_FMT, datefmt=LOG_DATETIME_FORMAT)
            if not only_file:
                self._logger.addHandler(self._get_console_handler())

            self._logger.addHandler(
                self._get_file_handler(filename=os.path.join(log_dir, log_file_name))
            )
            self._logger.setLevel(log_level)

    def _get_file_handler(self, filename):
        """返回一个文件日志handler"""
        file_handler = logging.FileHandler(filename=filename, encoding="utf8")
        file_handler.setFormatter(self.formatter)
        return file_handler

    def _get_console_handler(self):
        """返回一个输出到终端日志handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

    @property
    def logger(self):
        return self._logger


logger = Logger().logger
