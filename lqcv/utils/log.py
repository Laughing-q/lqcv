import logging.config

def set_logging(name="lqcv", verbose=True, format="%(message)s"):
    level = logging.INFO if verbose else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': f"{format}"}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False}}})
    return 

set_logging()
LOGGER = logging.getLogger('lqcv')
