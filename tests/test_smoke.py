"""Smoke test: confirms the package imports and pytest discovery works in CI."""

import sensoryaware


def test_package_has_version():
    assert sensoryaware.__version__ == "0.1.0"
