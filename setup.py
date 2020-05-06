#!/usr/bin/env python
"""NiWorkflows setup script."""
import sys
from setuptools import setup
import versioneer

setupargs = {
    "version": versioneer.get_version(),
    "cmdclass": versioneer.get_cmdclass(),
}

# Give setuptools a hint to complain if it's too old a version
# 30.3.0 allows us to put most metadata in setup.cfg
# 30.4.0 gives us options.packages.find
# 34.0.2 ensure extras are honored (e.g., for console_scripts)
# 36.3 Several files within file: directive in metadata.long_description
# 36.4 Description-Content-Type medatada, misc fixes
# 36.7 Support setup_requires in setup.cfg files.
# 38.0 More deterministic builds
# 38.2.2 fix handling of namespace packages when installing from a wheel.
# 38.3 Support for long_description_type in setup.cfg, PEP 345 Project-URL metadata
# Should match pyproject.toml and setup.cfg

if "bdist_wheel" in sys.argv:
    # This enables setuptools to install wheel on-the-fly
    setupargs["setup_requires"] = ["setuptools >= 38.3.0", "wheel"]

if __name__ == "__main__":
    setup(**setupargs)
