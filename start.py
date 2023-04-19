import sys
sys.path.append('./svm_model')

from typing import Optional
import typer

from svm_model.train_pipeline import run_training
from svm_model.predict import make_prediction
from svm_model.processing.data_manager import load_dataset

app = typer.Typer()

@app.command()
def train_mode(name: Optional[str] = None):
	run_training()

	print("Train Success.")

@app.command()
def predict_mode(name):
	multiple_test_input = load_dataset(file_name=name)
	subject = make_prediction(input_data=multiple_test_input)
	
	print(subject.get('predictions'))
	print("Predict Success.")

if __name__ == "__main__":
	app()