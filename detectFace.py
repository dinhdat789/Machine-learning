import cv2
import setting
import os

haar_face = cv2.CascadeClassifier(setting.haarcascade_path + "//haarcascade_frontalface_default.xml")

def DetectFace(src, saveSrc):
    list_folder = os.listdir(src)
    for folder in list_folder:
        list_file = os.listdir(os.path.join(src, folder))
        for ifile in list_file:
            img = cv2.imread(os.path.join(src, folder, ifile))
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = haar_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 3, minSize = (30, 30))
            _length = len(faces)
            if _length != 0 :
                for (x, y, w, h) in faces:
                    crop = img[y:y+h, x+5:x+w-5]
                    cv2.imwrite(os.path.join(saveSrc, folder) + "//" + folder + ifile, crop)

if __name__=="__main__":
    gender_path = setting.gender_path
    face_path = setting.face_path

    test_gender_path = setting.test_gender_path
    test_face_path = setting.test_face_path

    DetectFace(gender_path, face_path)

    DetectFace(test_gender_path, test_face_path)
