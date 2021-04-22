
.. include:: links.rst

Installation
============

Make sure all of *NIWorkflows*' `External Dependencies`_ are installed.
These tools must be installed and their binaries available in the
system's ``$PATH``.
A relatively interpretable description of how your environment can be set-up
is found in the `Dockerfile <https://github.com/nipreps/niworkflows/blob/master/Dockerfile>`_.
As an additional installation setting, FreeSurfer requires a license file ..

On a functional Python 3.5 (or above) environment with ``pip`` installed,
*NIWorkflows* can be installed using the habitual command ::

    $ python -m pip install niworkflows

Check your installation with the following command line ::

    $ python -c "from niworkflows import __version__; print(__version__)"


External Dependencies
---------------------

The *NIWorkflows* are written using Python 3.5 (or above), and is based on
nipype_.

*NIWorkflows* require some other neuroimaging software tools that are
not handled by the Python's packaging system (Pypi) used to deploy
the ``niworkflows`` package:

- FSL_ (version 5.0.9)
- ANTs_ (version 2.2.0 - NeuroDocker build)
- AFNI_ (version Debian-16.2.07)
- `C3D <https://sourceforge.net/projects/c3d/>`_ (version 1.0.0)
- FreeSurfer_ (version 6.0.1)
