import sqlite3
import tkinter as tk
from tkinter import *
from tkinter import font as tkfont
import datetime
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

class EmotrikaApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family="Brush Script MT",size=28)
        self.title("Emotrika")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=10)
        container.grid_columnconfigure(1, weight=10)
        container.grid_columnconfigure(1, minsize=100)
        self.geometry("%dx%d+%d+%d" % (800, 500, 500, 200))

        self.frames = {}
        for F in(StartScreen,SignupScreen,LoginScreen, MainScreen):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name]=frame
            frame.grid(row=0, column=1, sticky="nsew")
            frame.config(bg="#414a4c")

        self.show_frame("StartScreen")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class emotion_recognise():
    # parameters for loading data and images
    detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
    emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
     "neutral"]

    #feelings_faces = []
    #for index, emotion in enumerate(EMOTIONS):
       # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

    # starting video streaming
    cv2.namedWindow('your_face')
    camera = cv2.VideoCapture(0)
    while True:
        frame = camera.read()[1]
        #reading the frame
        frame = imutils.resize(frame,width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
                        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            global useremotion
            useremotion = label
        else: continue

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                    # draw the label + probability bar on the canvas
                    # emoji_face = feelings_faces[np.argmax(preds)]

                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                  (0, 0, 255), 2)
    #    for c in range(0, 3):
    #        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
    #        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
    #        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
 
class database():
    def create_table_logins():
        dbcon = sqlite3.connect("database.db")
        dbcon.execute("CREATE TABLE IF NOT EXISTS logins(userID int NOT NULL,username text,password text,email text,number text)")
        dbcon.commit()

    def insert_user():
        dbcon = sqlite3.connect("database.db")
        dbcon.execute("INSERT INTO logins(username, password) VALUES(?,?)",(userName2, passWord2))
        dbcon.commit()

    def create_table_emotions():
        dbcon = sqlite3.connect("database.db")
        dbcon.execute("CREATE TABLE IF NOT EXISTS emotions(username,datetime text,userEmotion text)")
        dbcon.commit()

    def insert_emotion():
        datetime=datetime.datetime.now()
        dbcon = sqlite3.connect("database.db")
        dbcon.execute("INSERT INTO emotions(username,datetime, userEmotion) VALUES(?,?,?)",(userName2,datetime,useremotion))
        dbcon.commit()

    def get_user_data():
        dbcon = sqlite3.connect("database.db")
        dbcon.execute("SELECT * from emotions WHERE username = ?",(userName2))
        dbcon.commit()
        
class StartScreen(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label2 = tk.Label(self, text="Welcome to Emotrika", font=controller.title_font, bg="#272b2e", fg="white", height=2)
        label2.pack(fill=X, pady=30)
        
        label1 = tk.Label(self, text="Your Mental Health Companion", font=controller.title_font, bg="#D3D3D3", fg="black")
        label1.pack()

        
        
        button1 =tk.Button(self, text="Let's Go", bg="#aec5e0", fg="white",height=2, width=20,
                             command=lambda: controller.show_frame("LoginScreen"))
        button1.pack(pady=100)
        

class SignupScreen(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        def sign_check(event=None):
                global userName2, passWord2
                userName2 = userName.get()
                passWord2 = passWord.get()
                if not userName2 or not passWord2:
                    print(userName2)
                    lbl_text.config(text="Please complete the required field!", fg="red")
                else:
                    database.create_table()
                    ("Username: %s Password: %s" % (userName.get(), passWord.get()))
                    database.insert_user()
                    controller.show_frame("LoginScreen")
     
        label1 = tk.Label(self, text="Create An Account:", font=controller.title_font, bg="#272b2e", fg="white")
        label1.pack(pady=20)

        label2 = tk.Label(self, text="Username",bg="#272b2e", fg="white",height=1, width=25)
        label2.pack(pady=20)
        userName = Entry(self,bg="#26466D",fg="white",width=20)
        userName.pack()
        
        label3 = tk.Label(self, text="Password",bg="#272b2e", fg="white",height=1, width=25)
        label3.pack(pady=20)
        passWord = Entry(self,bg="#26466D",fg="white",width=20)
        passWord.pack(pady=20)

        button3 = tk.Button(self, text="Accept",bg="#aec5e0", fg="white",
                            command=lambda: controller.show_frame("LoginScreen"))
        button3.pack()

        lbl_text = Label(self)
        lbl_text.config(bg="white")
        lbl_text.pack()
        
        button1 =tk.Button(self, text="Log in ⇛ ",bg="#5190ED", fg="white", height=3, width=30,
                             command=sign_check)
        button1.pack(side="right", padx=10, pady=20)

class LoginScreen(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label1 = tk.Label(self, text="Login with existing account:", font=controller.title_font, bg="#272b2e", fg="white")
        label1.pack(pady=20)

        label2 = tk.Label(self, text="Username",bg="#272b2e", fg="white",height=1, width=25)
        label2.pack(pady=20)
        global userName
        userName = Entry(self,bg="#26466D",fg="white",width=20)
        userName.pack()
        
        label3 = tk.Label(self, text="Password",bg="#272b2e", fg="white",height=1, width=25)
        label3.pack(pady=20)
        global passWord 
        passWord = Entry(self,bg="#26466D",fg="white",width=20)
        passWord.pack(pady=20)

        def account_check():
            ("Username:%s Password:%s" % (userName.get(), passWord.get()))
            
            global userName3, passWord3
            userName3 = userName.get()
            passWord3 = passWord.get()
            account_exist()

        def account_exist():                                 
                dbcon = sqlite3.connect("database.db")
                c = dbcon.cursor()
                values = (userName3, passWord3)
                sql = """SELECT * FROM logins WHERE username=? AND password=?"""
                c.execute(sql, values)
                exists = c.fetchone()
                if not exists:
                        print("Username or Password not found... Please Try Again")
                else:
                        controller.show_frame("MainScreen")
        
        button3 = tk.Button(self, text="Accept",bg="#aec5e0", fg="white",
                            command=account_check)
        button3.pack()
        
        button1 =tk.Button(self, text="Sign Up ⇛ ",bg="#5190ED", fg="white", height=3, width=30,
                             command=lambda: controller.show_frame("SignupScreen"))
        button1.pack(side="right", padx=10, pady=20)
        
class MainScreen(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label1 = tk.Label(self, text="Emotrika Home", font=controller.title_font, bg="#272b2e", fg="white")
        label1.pack()

        button1 = tk.Button(self, text="New recording",bg="#aec5e0", fg="white", height=3, width=30,
                            command=emotion_recognise)
        button1.pack(padx=10, pady=20)

        button2=tk.Button(self, text="Previous Recordings",bg="#aec5e0", fg="white", height=3, width=30,
                            command=database.get_user_data)
        button2.pack(padx=10, pady=20)

        button3 = tk.Button(self, text="Exit",bg="#aec5e0", fg="white", height=3, width=30,
                            command=root.destroy())
        
if __name__ =="__main__":
    app = EmotrikaApp()
    app.mainloop()
