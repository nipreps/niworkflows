
.PHONY: clean
clean: clean-pyc clean-build

.PHONY: clean-pyc
clean-pyc:
		find . -name '__pycache__' -type d -exec rm -r {} +
		find . -name '*.pyc' -exec rm --force {} +
		find . -name '*.pyo' -exec rm --force {} +
		find . -name '*~' -exec rm --force  {} +

.PHONY: clean-build
clean-build:
		rm --force --recursive build/
		rm --force --recursive dist/
		rm --force --recursive .egg/
		rm --force --recursive *.egg-info
		rm --force --recursive src/

dist: clean-build clean-pyc
		python setup.py sdist
