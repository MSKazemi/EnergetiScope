.DEFAULT_GOAL := help

##@ Setup
.PHONY: install
install: ## Install project dependencies
	uv sync

.PHONY: install-dev
install-dev: ## Install project with dev dependencies
	uv sync --extra dev

##@ Development
.PHONY: train
train: ## Train the power prediction model
	uv run python app/train_power.py

.PHONY: predict
predict: ## Run the prediction service
	uv run python app/predict_service.py

.PHONY: collect
collect: ## Run the Kubernetes data collector
	uv run python app/k8s_collect.py

##@ Quality
.PHONY: test
test: ## Run tests
	uv run pytest

.PHONY: lint
lint: ## Run linter
	uv run ruff check .

.PHONY: format
format: ## Format code
	uv run ruff format .

##@ Evaluation
.PHONY: eval
eval: ## Run evaluation on all datasets
	uv run python eval/eval_all.py

##@ Docker
.PHONY: docker-build
docker-build: ## Build Docker image
	docker build -t energetiscope .

##@ Help
.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
