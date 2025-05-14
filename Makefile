install:
	pip install -r requirements.txt

train:
	python src/train.py

evaluate:
	python src/evaluate.py

batch-score:
	python src/batch_score.py

serve:
	uvicorn src.real_time_api:app --host 0.0.0.0 --port 8000

mlflow-ui:
	mlflow ui --port 5000
