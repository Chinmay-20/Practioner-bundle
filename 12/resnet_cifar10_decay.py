#python resnet_cifar10_decay.py --output output --model output/resnet_cifar10.hdf5

import matplotlib

matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from resnet import ResNet
from trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as no
import argparse
import sys
import os

sys.setrecursionlimit(5000)

NUM_EPOCHS=100
INIT_LR=1e-1

def poly_decay(epoch):
	maxEpochs=NUM_EPOCHS
	baseLR=INIT_LR
	power=1.0
	
	alpha=baseLR*(1-(epoch/float(maxEpochs)))**power
	
	return alpha
	
	
ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help="path to output model")
ap,add_argument("-o","--output",required=True, help="path to output directory(logs,plots,etc)")
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

figPath=os.path.sep.join([args["output"],"{}.png".format(os.getpid())])
jsonPath=os.path.sep.join([args["output"],"{}.json".format(os.getpid())])

callbacks=[
	TrainingMonitor(figPath,jsonPath=jsonPath),
	LearningRateScheduler(poly_decay)]
	
	
print("[INFO] compiling model...")
opt=SGD(lr=INIT_LR,momentum=0.9)

model=ResNet.build(32,32,3,10,(9,9,9),(64,64,128,256),reg=0.0005)
model.compile(loss="categorical_crrossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network....")
model.fit(aug.flow(trainX,trainY,batch_size=128),validation_data=(testX,testY),steps_per_epoch=len(trainX)//128,epochs=10, callbacks=callbacks,verbose=1)

print('[INFO] serialling network...")
model.save(args["model"])
