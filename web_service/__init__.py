import json
from flask import Flask, Response, request, jsonify

from .driver import Driver
from .template import TrainResponseTemplate, AnalyzeResponseTemplate
from .lib.FastTextPreprocessor import FastTextPreprocessor

app = Flask(__name__)
white_listed_ip = {
	"192.168.0.254",
	"127.0.0.1"
}

@app.route("/")
def index():
	response_obj = TrainResponseTemplate()

	if request.remote_addr in white_listed_ip:
		response_obj.success = True
		response_obj.message = "It Works!"
	else:
		response_obj.success = False
		response_obj.message = request.host# "Forbidden"

	# response_obj.success = True	
	# response_obj.message = "It Works!"
	response = json.dumps(response_obj, default=jdefault)

	return Response(response), 200

@app.route("/analyze", methods=['POST'])
def analyze():
	# Retrieving Inputs
	src = request.form.get('src')
	title = request.form.get('title')
	article = request.form.get('article')

	# Analyzing 
	prob = Driver.predict_text(src=src, title=title, article=article)

	# Configuring Response
	response_obj = AnalyzeResponseTemplate()
	response_obj.success = True
	response_obj.real_prob = prob
	response = json.dumps(response_obj, default=jdefault)

	return app.response_class(
		response=response,
		status=200,
		mimetype='application/json')

@app.route("/add_dataset", methods=['POST'])
def add():
	# Retrieving Inputs
	src = request.form.get('src')
	title = request.form.get('title')
	article = request.form.get('article')
	label = request.form.get('label')

	# Insert to Dataset
	Driver.insert_dataset(src=src, title=title, article=article, label=label)
	return Response("Done"), 200

@app.route("/train", methods=['POST'])
def train():
	# Performing Training
	Driver.train_model()

	# Configuring Response
	# response_obj = TrainResponseTemplate()
	return Response("Done"), 200

	# json.dumps(object.__dict__)

def jdefault(o):
	return o.__dict__

if __name__ == "__main__":
	# app.run(debug=True)
	app.run(host='0.0.0.0:8000#')
