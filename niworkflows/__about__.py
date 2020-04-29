# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Package metadata."""
from datetime import datetime
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__packagename__ = "niworkflows"
__copyright__ = "Copyright {}, Center for Reproducible Neuroscience, Stanford University".format(
    datetime.now().year
)
__credits__ = [
    "Oscar Esteban",
    "Ross Blair",
    "Shoshana L. Berleant",
    "Christopher J. Markiewicz",
    "Chris Gorgolewski",
    "Russell A. Poldrack",
]
