import pytest


@pytest.fixture(scope="module")
def tmp_path(tmp_path_factory):
    """Fixture to create a temporary directory shared across tests."""
    yield tmp_path_factory.mktemp("shared")


@pytest.fixture(scope="module")
def deduplicate_output_path(tmp_path):
    return tmp_path / "output.smi"


@pytest.fixture(scope="module")
def split_output_dir(tmp_path):
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    return split_dir


@pytest.fixture(scope="module")
def processed_output_dir(tmp_path):
    return tmp_path / "processed"


@pytest.fixture(scope="module")
def augmented_output_dir(tmp_path):
    return tmp_path / "augmented"
