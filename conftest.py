import pytest


def pytest_collection_modifyitems(items):
    slow = pytest.mark.slow
    for item in items:
        if item.name.startswith("test_julia"):
            item.add_marker(slow)
