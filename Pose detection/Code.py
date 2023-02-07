# PRINCE KUMAR 


import cv2 as cv 
# importing cv2 library as cv
import mediapipe as mp 
# importing mediapipe as mp
import numpy as np 
# importing numpy as np




# initializing mediapipe pose solution
mp_pose  =  mp.solutions.pose      
mp_draw =   mp.solutions.drawing_utils
pose = mp_pose.Pose()



#load the video in cap
cap = cv.VideoCapture("michale.mp4")  
 



# read each frame/image from capture object
while True:
   ret,img =  cap.read()  
    

   # do Pose detection
   result = pose.process(img)
  
   # drawing the landmark and connecting them 
   mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
   mp_draw.DrawingSpec((255, 0, 0), 2, 2),mp_draw.DrawingSpec((255, 0, 255), 2, 3))

   #print the value of landmarks
   print(result.pose_landmarks)
    
   
   #fetching the dimension of video 
   h,w,c = img.shape
   


   # creating a white screen of same size as captured video 
   img1 = np.zeros([h,w,c])
   img1.fill(255)

   # drawing the landmark and connecting them 
   mp_draw.draw_landmarks(img1, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
   mp_draw.DrawingSpec((255, 0, 0), 2, 2),mp_draw.DrawingSpec((255, 0, 255), 2, 3))
   

   cv.imshow("Extracted pose",img1)
   # display the extracted image
   

   cv.imshow("Captured video",img)
   # display the capture video
   k = cv.waitKey(1)

   if ord("q")==k:
       break
cap.release()


      
