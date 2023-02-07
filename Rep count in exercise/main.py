# PRINCE KUMAR
import time
import cv2
# importing cv2 library as cv
import mediapipe as mp
# importing mediapipe as mp
import numpy as np
# importing numpy as np



# initializing mediapipe pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


# Create Function to calculate Angle by taking start mid and end point
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

# initialize the video reading '0' for web-cam
cap = cv2.VideoCapture("KneeBendVideo.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
size = (854, 640)
# counter variables buffer frame and stage
counter = 0
stage = None
ans = []
writer = cv2.VideoWriter(
    'output.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        y = time.time()
        ret, frame = cap.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        size = (frame_width, frame_height)
        ##########################################
        # store the frame to match the current frame in order to deal with fluctuation
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(frame, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            # matches1 = bf.match(des1, ans[-2])
            matches2 = bf.match(des1, ans[-2])
            matches3 = bf.match(des1, ans[-1])
        except:
            ans.append(des1)
        #########################################3
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)

            # knee counter logic
            if angle < 140 and stage == 'down':
                prev = time.time()
                stage = "up"
                if len(matches3) > 120 and len(matches2) > 120:
                    counter += 1

            if angle > 165:
                stage = "down"
                curr = time.time()
                if curr - prev < 8:
                    cv2.putText(image, "Keep your knee bent",
                                tuple(np.multiply(
                                    knee, [60, 300]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,
                                                              230, 230), 2, cv2.LINE_AA)

        except:
            pass

        # Render knee counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (100, 200, 100), -1)

        # Rep data
        cv2.putText(image, 'Knee', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(
                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )


        cv2.imshow('Exercise', image)
        
        writer.write(image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
