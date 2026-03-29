ENV_NAME := cmb-forecast

.PHONY: install test explore clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create conda env and install package (editable)
	conda env create -f environment.yml || conda env update -f environment.yml
	conda run -n $(ENV_NAME) pip install -e .

test:  ## Run the test suite
	conda run -n $(ENV_NAME) python -m pytest tests/ -v

explore:  ## Run the design exploration script (plots -> plots/)
	conda run -n $(ENV_NAME) python scripts/explore_designs.py

clean:  ## Remove the conda environment
	conda env remove -n $(ENV_NAME) -y
