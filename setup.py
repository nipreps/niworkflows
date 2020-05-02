#!/usr/bin/env python
"""niworkflows setup script."""
import sys
from setuptools import setup
import versioneer

if __name__ == '__main__':
    setupargs = {
        "version": versioneer.get_version(),
        "cmdclass": versioneer.get_cmdclass(),
    }
    if "bdist_wheel" in sys.argv:
        setupargs["setup_requires"] = ["setuptools >= 38.3.0", "wheel"]
    setup(**setupargs)
