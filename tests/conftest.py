import pytest

from nhs_context_logging import app_logger, logging_context

# noinspection PyUnresolvedReferences
from nhs_context_logging.fixtures import *  # noqa: F403
from nhs_context_logging.logger import uuid4_hex_string


@pytest.fixture(scope="session", autouse=True)
def global_setup():
    app_logger.setup("pytest", internal_id_factory=uuid4_hex_string)


@pytest.fixture(autouse=True)
def reset_logging_storage():
    logging_context.thread_local_context_storage()


@pytest.fixture
def unlock_global_setup():
    # Enable the app logger to be set up again
    app_logger._is_setup = False

    # Yield a window in which app logger modifications can occur
    yield

    # Revert to the global settings again
    app_logger._is_setup = False
    app_logger.setup("pytest", internal_id_factory=uuid4_hex_string)
