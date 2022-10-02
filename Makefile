SHELL=/bin/bash

ENV_NAME=nodalThermalSim

module_dir = $(shell dirname $(PWD))/module

CONDA_ACTIVATE=set -e; \
	       source $$(conda info --base)/etc/profile.d/conda.sh; \
	       conda activate $(ENV_NAME)

# configure conda
config:
	conda config --add channels conda-forge
	conda config --set channel_priority strict

# create the anaconda environment
create: config
	conda create --yes --name $(ENV_NAME) python=3.7 pip conda-build
	$(CONDA_ACTIVATE); \
	git config --global credential.helper cache

# all the following targets shall be executed after having activated an environment
#
# prepare the anaconda environment with the dev requirements
environment:
	pip install --requirement requirements.txt

# install lbvalid with its requirements
requirements:
	conda install --yes $$(cat requirements.txt)

# configure channel to get cs-package
channel:
	conda config --append channels file://$(module_dir)

# install for development
install: channel requirements
	pip install --force-reinstall --no-deps --editable .
#	conda install ../module/pytest_lbsolver-0.3.0-0.tar.bz2
#	conda install ../module/post_valid-1.1-py_0.tar.bz2
#	conda install ../module/antares-1.15.0-py_0.tar.bz2


.PHONY: doc
doc:
	cd doc && make html

# directory where the files used by the ci are placed
ci-dir:
	rm -rf .ci
	mkdir -p .ci

test: ci-dir
	LC_ALL=C \
	pytest -v \
		--disable-warnings \
		--junitxml .ci/junit-tests-results.xml \
		--cov $(ENV_NAME) \
		--cov-report html \
		--cov-report xml:.ci/coverage.xml \
		tests

# TODO: mypy --txt-report .ci/mypy.log $(NAME) tests
mypy: ci-dir
	mypy $(ENV_NAME) tests

pylint: ci-dir
	pylint -f parseable $(ENV_NAME) tests | tee .ci/pylint.log

flake8: ci-dir
	flake8 --tee --output-file .ci/flake8.log $(ENV_NAME) tests

check: mypy flake8

black:
	black -t py37 --exclude _version.py tests $(ENV_NAME)

isort:
	isort --recursive --apply tests $(ENV_NAME)

style: isort black

