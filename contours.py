"""Inputs : -video capture of real-time frames (for now, for testing not real time)
   Outputs : -boundary of robot
             -centroid of robot
             -obstacles boundaries
             -corner locations ? -->polygon/visibility graph
             -map boundary
             -objective boundaries
             -cropped image with things for visualization


             -dilate obstacles
             -erode map boundary
             -find direction of the robot
             -visibility graph


"""


from cv2 import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from shapely.geometry import LineString
from shapely.geometry import Polygon 


def polygon(c):
    # approximate the contour
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.005 * peri, True)
    return approx


plt.ion()
fig = plt.figure()

capture = cv.VideoCapture('robot.mp4')
while True:
    isTrue, frame = capture.read() # Reads video frame by frame
    height, width = frame.shape[:2]

    # Thresholds found by getting the pixel colors on many points
    lower_blue = np.array([50,30,20])
    upper_blue = np.array([90,55,50])

    lower_red = np.array([80,50,165])
    upper_red = np.array([170,150,210])

    lower_green = np.array([50,85,35])
    upper_green = np.array([110,130,85])

    lower_white = np.array([150,150,120])
    upper_white = np.array([200,200,200])


    # Blurring to suppress noise
    blurred_frame = cv.GaussianBlur(frame, (5,5),0)
    mask_white = cv.inRange( blurred_frame, lower_white, upper_white)
    contours_white, _= cv.findContours(mask_white, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # Finding biggest white contour thanks to the area
    c = max(contours_white, key = cv.contourArea)
    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")



    # Perspective transformation matrix
    M = cv.getPerspectiveTransform(src_pts, dst_pts)

    # Directly warp the rotated rectangle to get the straightened rectangle
    blurred_frame = cv.warpPerspective(blurred_frame, M, (width, height))
    frame = cv.warpPerspective(frame, M, (width, height))

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().invert_yaxis()




    # Creating masks
    mask_blue = cv.inRange( blurred_frame, lower_blue, upper_blue)
    mask_green = cv.inRange( blurred_frame, lower_green, upper_green)
    mask_red = cv.inRange( blurred_frame, lower_red, upper_red)
    mask_white = cv.inRange( blurred_frame, lower_white, upper_white)

    # Finding contours for each mask
    contours_white, _ = cv.findContours(mask_white, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours_red, _ = cv.findContours(mask_red, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours_green, _ = cv.findContours(mask_green, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours_blue, _ = cv.findContours(mask_blue, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    c = max(contours_white, key = cv.contourArea)
    cv.drawContours(frame, c, -1, (255,255,255),3)


    for contour_red in contours_red:
        area = cv.contourArea(contour_red)

        if area > 8000:
            cv.drawContours(frame, contour_red, -1, (0,0,255),3)
            M = cv.moments(contour_red)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv.circle(frame, (cx, cy), 7, (0,0,255), -1)
                plt.plot(cx,cy,'ro')

           

    for contour_blue in contours_blue:
        area = cv.contourArea(contour_blue)

        if area > 8000:
            (x,y),radius = cv.minEnclosingCircle(contour_blue)
            center = (int(x),int(y))
            radius = int(radius+0.2*radius)
            cv.drawContours(frame,contours_blue, -1, [255,0,0],3)
            cv.circle(frame,center,radius,(255,0,0),2)
            cv.circle(frame, center, 7, (255, 0, 0), -1)
            plt.plot(x,y,'bo')


    for contour_green in contours_green:
        area = cv.contourArea(contour_green)

        if area > 8000:
           cv.drawContours(frame, contour_green, -1, (0,255,0),3)
           pts = np.squeeze(polygon(contour_green))
           pol = Polygon(pts)
           left_hand_side = pol.buffer(radius, join_style=3 ,single_sided=True)
           plt.plot(*left_hand_side.exterior.xy, 'g')
    
    pts_white = np.squeeze(polygon(c))
    pol_white = Polygon(pts_white)
    right_hand_side_white = pol_white.buffer(-radius, join_style=3 ,single_sided=True)
    plt.plot(*right_hand_side_white.exterior.xy, 'b')
           
    fig.canvas.draw()
    fig.canvas.flush_events()



    cv.imshow('Video',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break


capture.release()
cv.destroyAllWindows()