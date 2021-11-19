#python finetune_flowers17.py --dataset ../datasets/flowers17/images --model flowers17.model

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imagetoarraypreprocessor import ImgetoArrayPreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from fcheadnet import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-m","--model",required=True,hlp="path to output model")
args=vars(ap.parse_args())


aug=ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

print("[INFO] loading images...")
imagePaths=list(paths.list_images(args["dataset"]))
classNames=[pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames=[str(x) for x in np.unique(classNames)]


aap=AspectAwarePreprocessor(224,224)
iap=ImagetoArrayPreprocessor()

sdl=SimpleDatasetLoader(preprocessors=[aap,iap])
(data,labels)=sdl.load(imagePaths,verbose=500)
data=data.astype("float")/255.0

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25,random_state=42)

trainY=LabelBinarizer().fit_transform(trainY)
testY=LabelBinarizer().fit_transform(testY)

baseModel=VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

headModel=FCHeadNet.build(baseModel,len(classNames),256)

model=Model(inputs=baseModel.input,outputs=headModel)


for layer in baseModel.layers:
	layer.trainable=False
	
	
print("[INFO] compiling model...")
opt=RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] training model...")
model.fit(aug.flow(trainX,trainY,batch_size=32),validation_data=(testX,testY),epochs=25,steps_per_epoch=len(trainX)//32,verbose=1)


print("[INFO] evaluating after initialization...")
predictions=model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=classNames))

for layer in baseModel.layers[15:]:
	layer.trainable=True
	
	
print("[INFO] re-compiling model...")
opt=SGD(lr=0.001)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] fine-tuninig model...")
model.fit(aug.flow(trainX,trainY,batch_size=32),validaton_data=(testX,testY), epochs=100,steps_per_epoch=len(trainX)//32,verbose=1)

print("[INFO] evaluating after fine-tuning...")
predictions=model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=classNames))

print("[INFO] serializing model...")
model.save(args["model"])