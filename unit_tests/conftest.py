import pytest

from model import get_model


@pytest.fixture
def model_for_test():
    model = get_model()
    return model