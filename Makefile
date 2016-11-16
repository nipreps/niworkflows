
VERSION := $(shell python -c "import niworkflows; print niworkflows.info.__version__")

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
		rm --force --recursive *.egg-info
		rm --force --recursive src/

.PHONY: tag
tag:
		git tag -a $(VERSION) -m "Version ${VERSION}"
		git push origin $(VERSION)
		git push upstream $(VERSION)

.PHONY: test
test: clean-pyc
		py.test --ignore=src/ --verbose $(TEST_PATH)

dist: clean-build clean-pyc
		python setup.py sdist

.PHONY: tag-release
release: clean-build
		python setup.py sdist
		twine upload dist/*

.PHONY: tag-release
tag-release: clean-build tag
		python setup.py sdist
		twine upload dist/*

