#python train_model.py --db ../datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5 --model dogs_vs_cats.pickle

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import argparse
import pickle
import h5py

ap=argparse.ArgumentParser()
ap.add_argument("-d","--db",required=True,help="path HDF5 database")
ap.add_argument("-m","--model",required=True,help="path to output model")
ap.add_argument("-j","--jobs",type=int,default=-1,help="# of jobs to run when tuning hyperparameter")
args=vars(ap.parse_args())

db=h5py.File(args["db"],"r")
i=int(db["labels"].shape[0]*0.75)


print("[INFO] tuning hyperparameters...")
model=GridSearchCV(LogisticRegression(),params,cv=3,n_jobs=args["jobs"])
model.fit(db["feature"][:i],db["labels"][:i])
print("[INFO] best hyperparameters: {}".format(model.best_params_))

print("[INFO] evaluating...")
preds=model.predict(db["features"][i:])
print(classification_report(db["labels"][i:],preds,target_names=db["label_names"]))

acc=accuracy_score(db["labels"][i:],preds)
print("[INFO] score {}".format(acc))

print("[INFO] saving model...")
f=open(args["model"],"wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

db.close()
