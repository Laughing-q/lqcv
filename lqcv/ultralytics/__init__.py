try:
    import ultralytics
except:
    from lqcv.utils.log import LOGGER
    LOGGER.warning("Please install package `ultralytics` first!")
