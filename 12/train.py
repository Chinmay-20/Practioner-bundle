#python train.py --checkpoints output/checkpoints
#python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_25.hdf5 --start-epoch 25
#python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_35.hdf5 --start-epoch 35

import matplotlib
matplotlib.use("Agg")

import tiny_imagenet_config as config
from imagetoarraypreproccessor import ImagetoArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from epochcheckpoint import EpochCheckpoint
from trainingmonitor import TrainingMonitor
from hdf5datasetgenerator import HDF5DatasetGenerator
from resnet import ResNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
import keras.backend as K
import argparse
import json64,
import sys

sys.setrecursionlimit(5000)

ap=argaprse.ArgumentParser()
ap.add_argument("-c","--checkpoints",required=True,help="path to output checkpoint directory")
ap.add_argument("-m","--model",type=str,help="path to *specific* model checkpoint to load")
ap.add_argument("-s","--start-epoch",type=int,default=0,help="epoch to restart training at")
args=vars(ap.parse_args())

aug=ImageDataGenerator(rotation_range=18,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_model="nearest")
means=json.loads(open(config.DATASET_MEAN).read()

sp=SimplePreprocessor(64,64)
mp=MeanPreprocessor(means["R",means["G"],means["B"])
iap=ImagetoArrayPreprocessor()

trainGen=HDF5DatasetGenerator(config.TRAIN_HDF5,64,aug=aug,preproessors=[sp,mp,iap],classes=config.NUM_CLASSES)
valGen=HDF5DatasetGenerator(config.VAL_HDF5,64,preproessors=[sp,mp,iap],classes=config.NUM_CLASSES)

if args["model"] is None:
	print("[INFO] compiling model...")
	model=ResNet.build(64,64,3,config.NUM_CLASSES,(3,4,6),(64,128,256,512),reg=0.0005,dataset="tiny_imagenet")
	opt=SGD(le=1e-1,momentum=0.9)
	model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
else:
	print("[INFO] loading {}...".format(args["model"]))
	model=load_model(args["model"]
	
	print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr,1e-5)
	print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
	
callbacks=[
	EpochCheckpoint(args["checkpoints"],every=5,startAt=args["start_epoch"]),
	TrainingMonitor(config.FIG_PATH,jsonPath=config.JSON_PATH,startAt=args["start_epoch"])]

model.fit(trainGen.generator(),steps_per_epoch=trainGen.numImages//64,validation_data=valGen.generator(),validation_steps=valGen.numImages//64,epochs=50,max_queue_size=64*2,callbacks=callbacks,verbose=1)

trainGen.close()
valGen.close()


#02240919191
