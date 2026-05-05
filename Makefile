ENV_NAME := augr

.PHONY: install test validate-pico validate-bk clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:  ## Create conda env and install package (editable)
	conda env create -f environment.yml || conda env update -f environment.yml
	conda run -n $(ENV_NAME) pip install -e .

test:  ## Run the test suite
	conda run -n $(ENV_NAME) python -m pytest tests/ -v

validate-pico:  ## Reproduce the PICO published-sigma(r) cross-check
	conda run -n $(ENV_NAME) python scripts/validate_pico.py

validate-bk:  ## Reproduce the BICEP/Keck sigma(r) time evolution
	conda run -n $(ENV_NAME) python scripts/validate_bk.py

clean:  ## Remove the conda environment
	conda env remove -n $(ENV_NAME) -y
