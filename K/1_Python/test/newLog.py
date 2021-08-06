import logging

logging.basicConfig(
    filename = 'test.log',
)

# logger = logging.getLogger('main') # 새로운 로거생성
logger = logging.getLogger(__name__) # 일반적으로 모듈 별로 이름을 만든다
logger.setLevel(logging.INFO) # 새로거의 레벨 설정 

logging.info('root에 info 기록') # 루트에 기록 
logging.warning('root에 warning  기록')

logger.info('메인에서 info기록') # 새로만든 로거에 기록
logger.warning('메인에서 warning 기록')