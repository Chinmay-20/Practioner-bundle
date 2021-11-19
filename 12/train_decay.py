#python train_decay.py --model output/resnet_tinyimagenet_decay.hdf5 --output output
#python rank_accuracy.py

import matplotlib
matplotlib.use("Agg")


import tiny_imagenet_config as config
from imagetoarraypreprocessor import ImagetoArrayPreprocessor 
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
import json
import sys

NUM_EPOCHS=75
INIT_LR=1e-1

def poly_decay(epoch):
	maxEpochs=NUM_EPOCHS
	baseLR=INIT_LR
	power=1.0
	
	alpha=baseLR*(1-(epoch/float(maxEpochs)))**power
	
	return alpha
	
ap=argparse.ArgumentParser()
ap.add_argument("-m",,"--model",required=True,help="path to output model")
ap.add_argument("-o","--output",required=True,help="path to output directory (logs,plots,etc.)")
args=vars(ap.parse_args())

print("[INFO] serializing network...")
model.save(args["model"])

trainGen.close()
valGen.close()


