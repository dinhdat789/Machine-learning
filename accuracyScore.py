import cv2
import os
import numpy as np
import joblib
import setting
from sklearn.metrics import accuracy_score

clf = joblib.load(setting.model_path + "//model.joblib")

_embedder = cv2.dnn.readNetFromTorch(setting.open_face_path + "//openface_nn4.small2.v1.t7")

def DetectFeature(image):
    face_blob = cv2.dnn.blobFromImage(image, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    _embedder.setInput(face_blob)
    feature_vector = _embedder.forward()
    feature_vector = feature_vector.flatten()
    return feature_vector

def LoadFeature(src):
    print("[*] Loading data ...")
    testing_dataset = []
    testing_lable = []
    list_folder = os.listdir(src)
    for folder in list_folder:
        list_file = os.listdir(os.path.join(src, folder))
        for ifile in list_file:
            print("[+] Loading Folder: ", folder)
            _img = cv2.imread(os.path.join(src, folder, ifile))
            img_feature = DetectFeature(_img)
            testing_dataset.append(img_feature)
            testing_lable.append(folder)
    print("[*] Load Data Finished")
    testing_lable = np.array(testing_lable)
    return testing_dataset, testing_lable

if __name__=="__main__":
    src = setting.test_face_path
    testing_dataset, testing_lable = LoadFeature(src)
    result = clf.predict(testing_dataset)
    print("Accuracy: %.2f %%" %(100*accuracy_score(testing_lable, result)))