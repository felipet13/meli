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
	pip install -r src/requirements.txt && \
	pre-commit install
	
@Echo Song
 "Environment $(ENV_NAME) successfully created"
install-clean:
	conda remove -n $(ENV_NAME) --all -y
	
@Echo Song
 "Environment $(ENV_NAME) successfully removed"
	conda create --name $(ENV_NAME) -y python=$(PYTHON_VERSION) --force
	$(CONDA_ACTIVATE) $(ENV_NAME) && \
	pip install -r src/requirements.txt && \
	pre-commit install
	
@Echo Song
 "Environment $(ENV_NAME) successfully created"
update:
	$(CONDA_ACTIVATE) $(ENV_NAME) && \
	pip install -r src/requirements.txt -U
test:
	python -m pytest
lint:
	${CONDA_ACTIVATE} $(ENV_NAME)
	pre-commit run -a
prefect-start:
	prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
	prefect server start
prefect-agent:
	prefect agent start --pool "default-agent-pool" --work-queue "default" --prefetch-seconds 300 -q default
prefect-deploy-prd:
	python orchestration/prefect_flows/prd/register_pi_ingestion_flow.py
	python orchestration/prefect_flows/prd/register_live_recommend_flow.py
	python orchestration/prefect_flows/prd/register_pi_ingestion_backfill_flow.py
	python orchestration/prefect_flows/prd/register_cra_status_flow.py
	python orchestration/prefect_flows/prd/register_export_data_flow.py
	python orchestration/prefect_flows/prd/register_export_implementation_flow.py
load-tag-dict-cra-prd:
	python utilities/cra_upload_tagdict.py --cra_api=http://172.26.0.11
load-tag-dict-cra-local:
	python utilities/cra_upload_tagdict.py --cra_api=http://localhost