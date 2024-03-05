import logging


logging.basicConfig(
    level=logging.INFO,
    filename='test_platform.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
