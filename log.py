import os
import sys
import logging
import time

level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def setLevel(logger, level='info'):
    '''
    level: debug, info, warning, error, critical
    '''
    logger.setLevel(level_map[level])

def getLogger(name, mode='file', dirname=None,
              level=logging.INFO,
              fmt_msg='%(asctime)s [%(module)s:%(lineno)d - %(levelname)s] %(message)s',
              fmt_date='%Y-%m-%d %H:%M:%S'):
    """
    mode: configure the output target the print message will be sent to.
    if set with `file` the message will be written to the `name` file under `dirname` directory; otherwise, sent to the console
    level: message that level under `level` parameter will be ignored.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter(fmt=fmt_msg, datefmt=fmt_date)

    if mode == 'file':
        if dirname is None:
            dirname = os.path.join(os.getcwd(), "logs")
        else:
            dirname = os.path.join(dirname, "logs")

        if not os.path.exists(dirname):
            os.mkdir(dirname)

        filename = os.path.join(dirname, time.strftime('%Y-%m-%d-') + name + '.log')
        if os.path.exists(filename):
            os.remove(filename)

        handler = logging.FileHandler(filename)
    elif mode == "console":
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(fmt)
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    logger = getLogger('test', mode="file")
    setLevel(logger, "debug")

    logger.debug("hello world")
