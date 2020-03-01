import cv2
import numpy as np
import random
video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
img_1=cv2.imread("black.png", cv2.IMREAD_COLOR)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
x=0
y=0
w=0
h=0
yf=0
h1=[160,180,200,180,160,140]
w1=[20,20,20,20,20,20]
h2=[160,180,200,180,160,230]
w2=[20,20,20,20,20,20]
xo=[200,300,400,500,600,700]
yo=[0,0,0,0,0,0,0,0,0]
k=yf
for i in range (0,100):
    _, first_frame = video.read(0)
    img_1=cv2.resize(img_1,(int(first_frame.shape[1]),int(first_frame.shape[0])))
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    cv2.waitKey(5)  
    for (a,b,c,d) in faces: 
        cv2.rectangle(first_frame,(a,b),(a+c,b+d),(255,255,0),2)
        x=a
        y=b
        w=c
        h=d
        #yf=a
    if(x!=0 and y!=0 and w!=0 and h!=0 and i>10):
        break
    
    cv2.imshow("ye",first_frame)
yf=100
xcpy=x
wcpy=w
cv2.destroyAllWindows() 
print("GAME IS STARTING")
cv2.waitKey(1000)
roi = first_frame[y: y + h, x: x + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
while True:
    _, frame = video.read()
    #frame=frame[0:frame.shape[0], (xcpy-100):(xcpy+100+wcpy)]
    #frame=frame[(x-40):0,(x+w+40):frame.shape[0]]
    #cv2.imshow("CCC",frame)
    #cv2.waitKey(7000)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    _, track_window = cv2.meanShift(mask, (x, y, w, h), term_criteria)
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    #cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    
    #cv2.rectangle(img_1,(0,0),(img_1.shape[1],60),(0,255,0),-2)
    #cv2.rectangle(img_1,(0,img_1.shape[0]-60),(img_1.shape[1],img_1.shape[0]),(0,255,0),-2)
    
    #cv2.waitKey(10000)
    
    #pixel2=img_1[10,10]
    #print(pixel2)
    #cv2.waitKey(10000)
    cv2.circle(img_1,(yf,y),20,(52,64,235),-1)
    flag=1
    for i in range(0,6):
        if((yf<(xo[i]+w1[i]) and yf>xo[i] and (y-20)<=(yo[i]+h1[i]))or(yf<(xo[i]+w2[i]) and yf>xo[i] and (y+20)>=(yo[i]+img_1.shape[0]-h2[i]) ) ):
            flag=0
            cv2.rectangle(img_1,(xo[i],yo[i]),(xo[i]+w1[i],yo[i]+h1[i]),(0,0,255),-1)
            cv2.rectangle(img_1,(xo[i],yo[i]+img_1.shape[0]-h2[i]),(xo[i]+w2[i],img_1.shape[0]),(0,0,255),-1)
        elif(((y-20)<=(yo[i]+h1[i]) and (yf+20)>=xo[i] and (yf+20)<=(xo[i]+w1[i]))or((y-20)<=(yo[i]+h1[i]) and (yf-20)<=(xo[i]+w1[i]) and (yf-20)>=xo[i])):
            flag=0
            cv2.rectangle(img_1,(xo[i],yo[i]),(xo[i]+w1[i],yo[i]+h1[i]),(0,0,255),-1)
            cv2.rectangle(img_1,(xo[i],yo[i]+img_1.shape[0]-h2[i]),(xo[i]+w2[i],img_1.shape[0]),(0,0,255),-1)
        elif(((y+20)>=(yo[i]+img_1.shape[0]-h2[i]) and (yf+20)>=xo[i] and (yf+20)<=(xo[i]+w1[i]))or((y+20)>=(yo[i]+img_1.shape[0]-h2[i]) and (yf-20)<=(xo[i]+w1[i]) and (yf-20)>=xo[i])):
            flag=0
            cv2.rectangle(img_1,(xo[i],yo[i]),(xo[i]+w1[i],yo[i]+h1[i]),(0,0,255),-1)
            cv2.rectangle(img_1,(xo[i],yo[i]+img_1.shape[0]-h2[i]),(xo[i]+w2[i],img_1.shape[0]),(0,0,255),-1)
        else:
            cv2.rectangle(img_1,(xo[i],yo[i]),(xo[i]+w1[i],yo[i]+h1[i]),(255,255,0),-1)
            cv2.rectangle(img_1,(xo[i],yo[i]+img_1.shape[0]-h2[i]),(xo[i]+w2[i],img_1.shape[0]),(0,255,255),-1)
    #cv2.imshow("ob",img_1)
    #poly = cv2.ellipse2Poly((yf,y),(40,40),0,0,360,1)
    #cv2.circle(img_1,(yf,y-35),4,(0,0,255),2)
    
    cv2.imshow("game",img_1)


    if(flag==0):
        cv2.waitKey(8000)
        break

    
    cv2.circle(img_1,(yf,y),20,(0,0,0),-1)
    for i in range(0,6):
        cv2.rectangle(img_1,(xo[i],yo[i]),(xo[i]+w1[i],yo[i]+h1[i]),(0,0,0),-1)
        cv2.rectangle(img_1,(xo[i],yo[i]+img_1.shape[0]-h2[i]),(xo[i]+w2[i],img_1.shape[0]),(0,0,0),-1)
        xo[i]-=3
        if(xo[i]<=0):
            xo[i]=img_1.shape[1]
            #w1[i]=random.randrange(30,60,1)
            h1[i]=random.randrange(150,250,1)
            #w2[i]=random.randrange(30,60,1)
            h2[i]=img_1.shape[0]-h1[i]-45-random.randrange(0,80,1)
    cv2.waitKey(1)
    key = cv2.waitKey(60)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()