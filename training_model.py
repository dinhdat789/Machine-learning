import cv2
import setting
import os
import joblib
import numpy as np
from sklearn.svm import LinearSVC

_embedder = cv2.dnn.readNetFromTorch(setting.open_face_path + "//openface_nn4.small2.v1.t7")

def DetectFeature(image):
    face_blob = cv2.dnn.blobFromImage(image, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    _embedder.setInput(face_blob)
    feature_vector = _embedder.forward()
    feature_vector = feature_vector.flatten()
    return feature_vector

def LoadFeature(src):
    print("[*] Loading data ...")
    training_dataset = []
    training_lable = []
    list_folder = os.listdir(src)
    for folder in list_folder:
        list_file = os.listdir(os.path.join(src, folder))
        for ifile in list_file:
            print("[+] Loading Folder: ", folder)
            _img = cv2.imread(os.path.join(src, folder, ifile))
            img_feature = DetectFeature(_img)
            training_dataset.append(img_feature)
            training_lable.append(folder)
    print("[*] Load Data Finished")
    training_lable = np.array(training_lable)
    return training_dataset, training_lable

def Clustering(trainingDataset, trainingLabel):
    print("[*] Clustering data ...")
    clf = LinearSVC()
    clf.fit(trainingDataset, trainingLabel)
    print("[*] Training Finished")
    return clf

def SaveModel(model):
    file_path = setting.model_path + "//model.joblib"
    print("[*] Saved model to file: ", file_path)
    joblib.dump(model, file_path)

if __name__ == "__main__":
    src = setting.face_path
    training_dataset, training_label = LoadFeature(src)
    clf = Clustering(training_dataset, training_label)
    SaveModel(clf)