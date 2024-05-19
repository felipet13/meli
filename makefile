ENV_NAME = meli
PYTHON_VERSION = 3.10
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
# Env name stored in environment.yml
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
## build and install all components and requirements
install:
	conda create --name $(ENV_NAME) -y python=$(PYTHON_VERSION) --force
	$(CONDA_ACTIVATE) $(ENV_NAME) && \
	pip install -r requirements.lock && \
	pre-commit install
	@echo "Environment $(ENV_NAME) successfully created"
install-clean:
	conda remove -n $(ENV_NAME) --all -y
	@echo "Environment $(ENV_NAME) successfully removed"
	conda create --name $(ENV_NAME) -y python=$(PYTHON_VERSION) --force
	$(CONDA_ACTIVATE) $(ENV_NAME) && \
	pip install -r requirements.lock && \
	pre-commit install
	@echo "Environment $(ENV_NAME) successfully created"
update:
	$(CONDA_ACTIVATE) $(ENV_NAME) && \
	pip install -r requirements.txt -U
test:
	python -m pytest
lint:
	${CONDA_ACTIVATE} $(ENV_NAME) && \
	pre-commit run -a
compile:
	pip-compile requirements.txt --output-file requirements.lock