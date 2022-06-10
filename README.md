# ml-fizz-buzz
Machine Learning approach to Fizz Buzz problem
- dataset.py - contains scripts for labeled dataset generation
- feature_preprocessors.py - custom feature preprocessors for feature engineering
- model.py - pipelines and code to train, dump and load model
- serving.py - model serving with FastAPI
- utils.py - some utilities
### Prerequisities
Create virtualenv and install requirements `pip install -r requirements.txt`
### Model training
For model training run `python model.py`
### Model serving
For model serving run `uvicorn serving:app --reload`

For endpoint auto-generated docs go to `http://localhost:8000/docs`
