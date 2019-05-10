Using PySpark Titanic Survival classifier is deployed using flask web app and expose a rest end point.

file information - 

1. server.py - which contain get and post reuest function, in get request training a model and saving it, in post request it load model and reponse output on requested data.
2. preprocess_file.py - which contain all preprocess function and transformation steps.
3. train_model.py - it contain model training and saving code.




Step 1 - Clone this repo in your system.
Step 2 - go to flask_app_deployment folder and run command - `export FLASK_APP=server`
Step 3 - run command - `flask run`
step 4 - open any rest client app ex. postman, ARC etc.

step 5 - hit url http://localhost:5000/         - it will start training and saving of model 
step 6 - hit url http://localhost:5000/predict   - this is post request so you need to pass data in json format.

Data request in json format for http://localhost:5000/predict

[{
	"PassengerId":894,
	"Pclass":1,
	"Name":"Kelly, Mrs. James",
	"Sex":"female",
	"Age":22,
	"SibSp":1,
	"Parch":1,
	"Ticket":"330911",
	"Fare":21,
	"Cabin":1,
	"Embarked":"S"
}]




