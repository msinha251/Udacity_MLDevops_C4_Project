import logging

def setup_logging(log_level="INFO"):
    """
        Set up logging for the application.
        :param log_level: The log level to use.
        :return: None
    """

    if log_level == 'DEBUG':
        print('Setting log level to DEBUG')
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        #logging.getLogger().setLevel(logging.DEBUG)
    elif log_level == 'INFO':
        print('Setting log level to INFO')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.getLogger().setLevel(logging.INFO)
    elif log_level == 'WARNING':
        print('Setting log level to WARNING')
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        #logging.getLogger().setLevel(logging.WARNING)
    elif log_level == 'ERROR':
        print('Setting log level to ERROR')
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        #logging.getLogger().setLevel(logging.ERROR)
    elif log_level == 'CRITICAL':
        print('Setting log level to CRITICAL')
        logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        #logging.getLogger().setLevel(logging.CRITICAL)

    return None

