import cv2
import sklearn
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageOps,ImageFilter
# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here
dataset_path = "D:\\Dataset_1\\images\\"

X = []
y = []

for i in glob.glob(dataset_path + '*.png',recursive = True):
    label = i.split("images")[1][1:4]
    y.append(label)
    image = cv2.imread(i)
    X.append(image)
    # write code to read ecah file i, and append it to list X
print(np.shape(X))
print(np.shape(y))

# you should have X, y with 5998 entries on each.
# T1 end ____________________________________________________________________________________


# T2 start __________________________________________________________________________________
# Preprocessing
X_processed = []
for x in X:
    # Write code to resize image x to 48x48 and store in temp_x
    temp_x = cv2.resize(x, (48, 48))
    # Write code to convert temp_x to grayscale
    temp_x = cv2.cvtColor(temp_x,cv2.COLOR_BGR2GRAY)
    # Append the converted image into X_processed
    X_processed.append(temp_x)
cv2.imshow('gray_scale',X_processed[0])
cv2.waitKey(0)

# T2 end ____________________________________________________________________________________


# T3 start __________________________________________________________________________________
# Feature extraction
X_features = []
for x in X_processed:
    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False, multichannel=False)
    X_features.append(x_feature)

# write code to Split training & testing sets using sklearn.model_selection.train_test_split
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X_features,y,test_size = 0.2)
#T3 end ____________________________________________________________________________________



#T4 start __________________________________________________________________________________
# train model
model = sklearn.svm.SVC()
model.fit(x_train,y_train)
pre = model.predict(x_test)
acc = model.score(x_test,y_test)
print(acc)