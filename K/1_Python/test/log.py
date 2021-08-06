import logging

logging.basicConfig(
    filename = 'test.log',
    level = logging.INFO,
)

logging.debug('이 메세지? 기록 안됨')
logging.info('이 메세지는? 기록 됨')
logging.error('이 메세지는? 기록 됨')