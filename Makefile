.PHONY: setup train train-tuned clean

setup:
	python -m venv venv
	venv\Scripts\pip install -r requirements.txt

train:
	cd src && python train.py --n_estimators 150 --max_depth 8

train-tuned:
	cd src && python train.py --grid_search

clean:
	del /Q models\*.pkl 2>nul
	del /Q outputs\*.json 2>nul
	if exist src\__pycache__ rmdir /S /Q src\__pycache__
	if exist __pycache__ rmdir /S /Q __pycache__