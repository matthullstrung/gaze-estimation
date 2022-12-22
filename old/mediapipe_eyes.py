import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
ret, img = cap.read()

# Last frame score
last_x = 0
last_y = 0

while(True):
    ret, img = cap.read()

    # To improve performance
    img.flags.writeable = False

    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Make it writeable again
    img.flags.writeable = True

    img_h, img_w, _ = img.shape
    
    if not results.multi_face_landmarks:
      continue

    for face_landmarks in results.multi_face_landmarks:

        # Eye center landmarks
        rcx,rcy, lcx,lcy = 0,0, 0,0
        
        # Eye corner landmarks
        inside_x, outside_x = 0,0
        
        # Eye top/bottom landmarks
        top, bottom = 0,0

        # Iterate over all landmark
        for idx, lm in enumerate(face_landmarks.landmark):
            # !!! ONLY WORKS FOR ONE EYE, FIXED FOR IN HEADPOSE.PY ITERATION !!!

            # Right iris center landmark
            if idx == 468:
                rcx,rcy = int(lm.x * img_w), int(lm.y * img_h)
                cv2.drawMarker(img, (rcx, rcy), (0,255,0), markerSize=5)

            # Left iris center landmark
            if idx == 473:
                lcx,lcy = int(lm.x * img_w), int(lm.y * img_h)
                cv2.drawMarker(img, (lcx, lcy), (0,255,0), markerSize=5) 


            # Eye inside corner landmark
            if idx == 133 or idx == 463:
                inside_x, inside_y = int(lm.x * img_w), int(lm.y * img_h)
                cv2.drawMarker(img, (inside_x, inside_y), (255,0,255), markerSize=5) 

            # Eye outside corner landmark
            if idx == 33 or idx == 359:
                outside_x, outside_y = int(lm.x * img_w), int(lm.y * img_h)
                cv2.drawMarker(img, (outside_x, outside_y), (255,0,255), markerSize=5)

            # Left/right Top eye landmark
            if idx == 159 or idx == 257:
                top_x, top = int(lm.x * img_w), int(lm.y * img_h)
                cv2.drawMarker(img, (top_x, top), (255,0,255), markerSize=5) 

            # Left/right Bottom eye landmark
            if idx == 145 or idx == 253:
                bottom_x, bottom = int(lm.x * img_w), int(lm.y * img_h)
                cv2.drawMarker(img, (bottom_x, bottom), (255,0,255), markerSize=5)


        # Calculate left eye scores (x,y)
        if (inside_x - outside_x) != 0:
            lx_score = (lcx - outside_x) / (inside_x - outside_x)
            if abs(lx_score - last_x) < .3: 
                cv2.putText(img, "x: {:.02}".format(lx_score), (lcx, lcy-30), cv2.FONT_HERSHEY_SIMPLEX, .25, (255,255,255), 1)
            last_x = lx_score
    
        if (bottom - top) != 0:
            ly_score = (lcy - top) / (bottom - top)
            if abs(ly_score - last_y) < .3: 
                cv2.putText(img, "y: {:.02}".format(ly_score), (lcx, lcy-20), cv2.FONT_HERSHEY_SIMPLEX, .25, (255,255,255), 1)
            last_y = ly_score


    # Show the image
    cv2.imshow('eyes', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()