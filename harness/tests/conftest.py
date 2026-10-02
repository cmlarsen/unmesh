import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs the full standard grid; enable with --slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="need --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
