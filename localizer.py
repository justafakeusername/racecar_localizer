import cv2
import numpy as np

print("hello world")

testing = 12


params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 3000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.9

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

doei = testing * 1

camera = cv2.VideoCapture(0)
print(camera.isOpened())
detector = cv2.SimpleBlobDetector_create(params)



for x in range(1000):
    f = None
    ret, image = camera.read(cv2.IMREAD_GRAYSCALE)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # cv2.imwrite("trial.png",image)    
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
    blobs = detector.detect(image)


    mask = cv2.inRange(hsv, (0, 150, 0), (255, 255,255))

    ## slice the green
    imask = mask>0
    green = np.zeros_like(image, np.uint8)
    green[imask] = image[imask]

    keypoints = detector.detect(green)

    im_with_keypoints = cv2.drawKeypoints(green, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    cv2.imshow("image", im_with_keypoints) 
    ## save 
    # cv2.imshow("green.png", gray)



    

    
    
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 20)

    # green = cv2.drawKeypoints(green, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imshow("green.png", green)

    cv2.waitKey(1)

