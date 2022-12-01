import cv2

#get face image 
face_img='human faces.webp'

# load some pre trained data on faces 
classifier_file1 ='haarcascade_frontalface_default.xml'

# load some pre trained data on smiles
classifier_file = 'haarcascade_smile.xml'

# face classifier
face_detector=cv2.CascadeClassifier(classifier_file1)

# smile classifier
smile_detector=cv2.CascadeClassifier(classifier_file)

# we cant provide much data on smile to algorithm as compared to the data we can provide 
# for faces , so to detect smile we have to give few more information to algorithm which will make it easier
# for the algorithm to detect smile  

# grab webcam feed
webcam = cv2.VideoCapture(0)

# show the current frame 
while True:

    # read the current frame from the webcam video stream
    successful_frame_read, frame = webcam.read() # this read function read a single frame 

    # if theres an error abort
    if not successful_frame_read :
        break
    
    # change to grayscale
    frame_grayscale= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     
    # detect faces 
    faces= face_detector.detectMultiScale(frame_grayscale)

    

    # draw rectangles 
    for(x,y,w,h) in faces:
       cv2.rectangle(frame, (x,y),(x+w,y+h),(100,200,50),4)
       
       # get the sub frame(using numpy N-dimensional array slicing)
       the_face=frame[y:y+h, x:x+w]

       # change to grayscale
       frame_grayscale= cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)
     

       # detect smiles
       smiles = smile_detector.detectMultiScale(frame_grayscale , scaleFactor=1.7, minNeighbors=20)
        #scalefactor this is an optimization to how much you want to blurr the image for making the detection easier
        # always we are going to use the same parameters to detect smiles 

        # find all the siles in the face
       #for(x_,y_,w_,h_) in smiles:

            # draw the rectangles around the smile
         #cv2.rectangle(the_face, (x_,y_),(x_ + w_, y_ + h_),(50,50,200),4)

       #instead of drawing the rectangle ariund the smile we are adding the text  
       if len(smiles)>0:
         cv2.putText(frame,'smiling',(x,y+h+40), fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))



    cv2.imshow('Why so serious',frame)

    #diplay
    cv2.waitKey(1) #agar main waitKey() mein kuch nhi daali tph merko key spam kra padega frame change krne ke liye
                   # par agar 1 daali hun toh after 1 millisecond woh automatically frame change krlega

#cleanup
webcam.release()
cv2.destroyAllWindows()

print("code completed")