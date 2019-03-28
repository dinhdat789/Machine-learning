import cv2
import numpy as np
import os
import joblib
import setting

img_path = input("Input image path: ")

img = cv2.imread(img_path)
newImg = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

clf = joblib.load(setting.model_path + "//model.joblib")

_embedder = cv2.dnn.readNetFromTorch(setting.open_face_path + "//openface_nn4.small2.v1.t7")
def DetectFeature(image):
    face_blob = cv2.dnn.blobFromImage(image, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    _embedder.setInput(face_blob)
    feature_vector = _embedder.forward()
    feature_vector = feature_vector.flatten()
    return feature_vector

haar_face = cv2.CascadeClassifier(setting.haarcascade_path + "//haarcascade_frontalface_default.xml")
faces = haar_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 3, minSize = (30, 30))
for (x, y, w, h) in faces:
    data = []
    face_img = img[y:y+h, x+5:x+w-5]
    feature = DetectFeature(face_img)
    data.append(feature)
    res = clf.predict(data)
    if res[0] == "female":
        cv2.rectangle(newImg, (x, y), (x+w, y+h), (0, 0, 255), 1)
    else:
        cv2.rectangle(newImg, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.putText(newImg, res[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow("result", newImg)
cv2.waitKey()
cv2.destroyAllWindows()

# cv2.imwrite("E://cccc.jpg", newImg)