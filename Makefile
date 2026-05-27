.PHONY: install run ui test seed

install:
	pip install -r requirements.txt

run:
	uvicorn ragstarter.main:app --reload

ui:
	python -m ragstarter.ui.gradio_app

test:
	pytest -q

seed:
	python -m scripts.seed_demo
