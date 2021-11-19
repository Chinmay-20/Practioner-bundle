#python resnet_cifar10.py --checkpoints output/checkpoints
#python resnet_cifar10.py --checkpoints output/checkpoints --model output/checkpoints/epoch_50.hdf5 --start-epoch 50
#python resnet_cifar10.py --checkpoints output/checkpoints --model output/checkpoints/epoch_75.hdf5 --start-epoch 75

import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from resnet import ResNet
from epochcheckpoint import EpochCheckpoint
from trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse
import sys

sys.setrecursionlimit(5000)

ap=Argparse.ArgumentParser()
ap,add_argument("-c","--checkpoints",required=True,help="path to output checkpoint directory")
ap.add_argument("-m","--model",type=str,help="path to *specific* model checkpoint to load")
ap.add_argument("-s","--start-epoch",type=int,default=0,help="epoch to restart training at")
args=vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((trainX,trainY),(testX,testY))=cifar10.load_data()
trainX=trainX.astype("float")
testX=testX.astype("float")

mean=np.mean(trainX,axis=0)
trainX-=mean
testX-=mean

lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)

aug=ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,fill_mode="nearest")

if args["model"] is None:
	print("[INFO] compiling model...")
	opt=SGD(lr=1e-1)
	model=ResNet.build(32,32,3,10,(9,9,9),(64,64,128,256),reg=0.0005)
	model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
else:
	print("[INFO] loading {}...".format(args["model"]))
	model=load_model(args["model"])
	
	print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
	
	K.set_value(model.optimizer.lr,1e-5) #this changes frequently as we do ctrl+c and again restart it
	
	print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
	
	
callbacks=[
	EpochCheckpoints(args["checkpoints"],every=5,startAt=args["start_epoch"]),
	TrainingMonitor("output/resnet56_cifar10.png",jsonPath="output/resnet56_cifar10.json",startAt=args["start_epoch"])]
	
print("[INFO] training network....")
model.fit(aug.flow(trainX,trainY,batch_size=128),validation_data=(testX,testY),steps_per_epoch=len(trainX)//128,epochs=10, callbacks=callbacks,verbose=1)






	
		
	
	
	
	
