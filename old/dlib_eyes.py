import cv2
import dlib
import numpy as np

# Convert landmarks from dlib model to an array of x and y coordinates
def get_landmarks(shape, dtype="int"):
	landmarks = np.zeros((68, 2), dtype=dtype)

	# Get x and y coordinates from landmark model
	for i in range(0, 68):
		landmarks[i] = (shape.part(i).x, shape.part(i).y)

	return landmarks

# Get iris center using moments
def get_iris_center(thresholded_img, mid, img, right=False):
    contours, _ = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cx, cy = 0,0
    try:
        max_contour = max(contours, key = cv2.contourArea)
        M = cv2.moments(max_contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (255, 0, 255), 2)
    except:
        pass

    return (cx,cy)

# Fill/mask eyes
def mask_eyes(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

# Eye parts
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresholded_img = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def callback(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, callback)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        
        # Get landmarks
        shape = predictor(gray, rect)
        shape = get_landmarks(shape)

        # Mask eyes
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = mask_eyes(mask, left)
        mask = mask_eyes(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

        # Threshold
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresholded_img = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresholded_img = cv2.erode(thresholded_img, None, iterations=2)
        thresholded_img = cv2.dilate(thresholded_img, None, iterations=4)
        thresholded_img = cv2.medianBlur(thresholded_img, 3)
        thresholded_img = cv2.bitwise_not(thresholded_img)

        # Find iris centers
        lcx, lcy = get_iris_center(thresholded_img[:, 0:mid], mid, img)
        rcx, rcy = get_iris_center(thresholded_img[:, mid:], mid, img, True)

        # Get left eye score
        lxscore = (lcx - shape[37][0]) / (shape[40][0] - shape[37][0])
        cv2.putText(img, "{:02}".format(lxscore), (lcx, lcy-20), cv2.FONT_HERSHEY_COMPLEX, .25, (255,255,255), 1)

        # Get right eye score
        rxscore = (rcx - shape[43][0]) / (shape[46][0] - shape[43][0])
        cv2.putText(img, "{:02}".format(rxscore), (rcx, rcy-20), cv2.FONT_HERSHEY_COMPLEX, .25, (255,255,255), 1)

    # Show image
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresholded_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()