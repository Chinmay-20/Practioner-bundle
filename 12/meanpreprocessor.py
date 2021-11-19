import cv2
 
class MeanPreprocessor:
	def __init__(self,rMean,gMean,bMean):
		self.rMean=rMean
		self.bMean=bMean
		self.gMean=gMean
		
	def preprocess(self,image):
		(B,G,R)=cv2.split(image.astype("float32"))
		
		R-=self.rMean
		G-=self.gMean
		b-=self.bMean
		
		return cv2.merge([B,G,R])
		
