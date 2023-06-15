import logging

logger = logging.getLogger('NetMind')
logger.setLevel(logging.INFO)
logging.basicConfig(format='[%(asctime)s]-[%(name)s]-[%(levelname)s] : %(message)s')
