from sklearn.neighbors import KNeighborsClassifier


import cv2
import pickle
import numpy as np 
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)



video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data\haarcascade_frontalface_default.xml')

with open('data/names.pkl' ,'rb') as f:
    LABELS=pickle.load(f)
with open('data/face_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)

imageBackground=cv2.imread("background.png")

COL_NAMES=['NAME','TIME']

while True:
    ret,frame=video.read()
    grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h,x:x+w, :]
        resized_img=cv2.resize(crop_img , (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist=os.path.isfile("Attendance/Attendance_"+ date +".csv")
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame,str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame,(x,y), (x+w,y+h),(50,50,255) , 1)
        attendance=[str(output[0]),str(timestamp)]
    imageBackground[162 : 162+480, 55 : 55+640]=frame
    cv2.imshow("Frame",imageBackground)
    k=cv2.waitKey(1)
    if k==ord('o'):
        speak("Attendance Taken")
        time.sleep(5)
        if exist:
            with open("Attendance/Attendance_"+ date +".csv" , "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_"+ date +".csv" , "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
            
    if k==ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()

from sklearn.model_selection import train_test_split

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(FACES, LABELS, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(knn, FACES, LABELS, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")



