import os
import cv2
import numpy as np

dataset_path="dataset/ECG_IMAGES_DATASET"
classes=os.listdir(dataset_path)

images=[]
labels=[]

for label,cls in enumerate(classes):
 folder=os.path.join(dataset_path,cls)
 for file in os.listdir(folder):
  path=os.path.join(folder,file)
  img=cv2.imread(path)
  img=cv2.resize(img,(224,224))
  img=img/255.0
  images.append(img)
  labels.append(label)

X=np.array(images)
y=np.array(labels)

print(X.shape)

np.save("X.npy",X)
np.save("y.npy",y)