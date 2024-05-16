import pytest


def pytest_addoption(parser):
    parser.addoption("--needsigblast", action="store_true", help="run igblast tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "needsigblast: mark test as needing igblast installation and databases")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--needsigblast"):
        # --needsigblast given in cli: do not skip igblast tests
        return
    skip_igblast = pytest.mark.skip(reason="need --needsigblast option to run")
    for item in items:
        if "needsigblast" in item.keywords:
            item.add_marker(skip_igblast)
