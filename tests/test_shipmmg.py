#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_shipmmg.

- version check
"""

from shipmmg import __version__


def test_version():
    """Test version."""
    assert __version__ == "0.0.10"
