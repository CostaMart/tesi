#
#       Main program to train / validate / test the models
#
#       Software is distributed without any warranty;
#       bugs should be reported to antonio.pecchia@unisannio.it
#


import mlflow as ml
from AutoEncoder import AutoEncoder
from mlflow import data
from mlflow.data import from_pandas
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from utils import evaluatePerformance, features, transformData

# MLFLow setup
ml.set_tracking_uri("http://localhost:5000")

ml.set_experiment("IDS_SemiSupervised")

train = 'data/TRAIN_AE.csv'
# train = 'C:/Users/Alessandro/deep_learning/hw4Data/TRAIN.csv'
validation = "data/VALIDATION.csv"
test = "data/TEST.csv"


X_tr, Y_tr, L_tr, X_val, Y_val, L_val, X_te, Y_te, L_te, trainSet, valSet, testSet = transformData(
    train, validation, test)

print(" training: " + str(X_tr.shape) + " ; " + str(Y_tr.shape))
print(" validat.: " + str(X_val.shape) + " ; " + str(Y_val.shape))

input_dim = X_tr.shape[1]


ids = AutoEncoder(input_dim=input_dim)
ids.TrainStepOne()
ids.summary()

with ml.start_run(run_name="testing_final_with_datset") as run:

    ids.train(X_tr, X_tr)

    outcome = ids.predict(X_val)

    recall, precision, f1, fpr = evaluatePerformance(outcome, L_val)

    ml.log_metric("validation_accuracy", recall)
    ml.log_metric("validation_precision", precision)
    ml.log_metric("validation_f1", f1)
    ml.log_metric("validation_fpr", fpr)

    ids.plot_reconstruction_error(X_val, L_val)
    outcome = ids.predict(X_te)

    evaluatePerformance(outcome, L_te)

    ml.log_metric("test_accuracy", recall)
    ml.log_metric("test_precision", precision)
    ml.log_metric("test_f1", f1)
    ml.log_metric("test_fpr", fpr)

    hyperparams = {"layers": 3, "sizes": [81, 64, 32, 16, 2], "types": ["input", "dense", "desne", "dense", "dense"],
                   "activation": ["relu", "relu", "relu", "tanh"], "epochs": 64, "batch_size": 64,
                   "scaler": "MaxAbsScaler", "optimizer": "Adam", "loss": "huber"}

    ml.log_params(hyperparams)
    sign = infer_signature(X_tr, outcome)

    trainSet[features].values.astype(float)

    trainSet = from_pandas(trainSet, source="myDictionary")

    ml.log_input(dataset=trainSet, context="train")

    model_info = ml.keras.log_model(model=ids.autoencoder, name="ids_autoencoder2",
                                    signature=sign, registered_model_name="ids_autoencoder2", params=hyperparams, model_type="autoencoder")

'''
ids = FeedforwardNN(input_dim=input_dim)
ids.summary()

ids.train(X_tr, Y_tr)

outcome = ids.predict(X_val)
evaluatePerformance(outcome, L_val)

outcome = ids.predict(X_te)
evaluatePerformance(outcome, L_te)
'''
