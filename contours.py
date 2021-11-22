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

# Test if it is the first frame of video
is_frame_one = 1

plt.ion()
fig = plt.figure()

polygon_obstacle_list = []
map_boundary = []
target_list = []
start = []



capture = cv.VideoCapture('robot.mp4')
while True:
    isTrue, frame = capture.read() # Reads video frame by frame

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
    if is_frame_one == 1:
        mask_white = cv.inRange( blurred_frame, lower_white, upper_white)
        contours_white, _= cv.findContours(mask_white, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        c = max(contours_white, key = cv.contourArea)
        x_rect,y_rect,w_rect,h_rect = cv.boundingRect(c)

    blurred_frame = blurred_frame[y_rect:y_rect+h_rect,x_rect:x_rect+w_rect]
    frame = frame[y_rect:y_rect+h_rect,x_rect:x_rect+w_rect]

    plt.xlim(0,w_rect)
    plt.ylim(0, h_rect)
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
            M = cv.moments(contour_red)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            if is_frame_one == 1 :
                plt.plot(cx,cy,'ro')
                target_list.append([cx,cy])
            cv.circle(frame, (cx, cy), 7, (0,0,255), -1)
            cv.drawContours(frame, contour_red, -1, (0,0,255),3)

            

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
            if is_frame_one == 1:
                start.append(center)
            


    for contour_green in contours_green:
        area = cv.contourArea(contour_green)
        if is_frame_one == 1 :
            if area > 8000:
                pts = np.squeeze(polygon(contour_green))
                pol = Polygon(pts)
                dilated_obstacle = pol.buffer(radius, join_style=3 ,single_sided=True)
                plt.plot(*dilated_obstacle.exterior.xy, 'g')
                polygon_obstacle_list.append(dilated_obstacle.exterior.xy)
        if area > 8000:
             cv.drawContours(frame, contour_green, -1, (0,255,0),3)



    if is_frame_one == 1 :
        pts_white = np.squeeze(polygon(c))
        pol_white = Polygon(pts_white)
        dilated_map = pol_white.buffer(-radius, join_style=3 ,single_sided=True)
        plt.plot(*dilated_map.exterior.xy, 'b')
        map_boundary.append(dilated_map.exterior.xy)
    
    is_frame_one = 0
    fig.canvas.draw()
    fig.canvas.flush_events()


    cv.imshow('Video',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
    

capture.release()
cv.destroyAllWindows()
plt.close()
print('targets :' , target_list)
print('start : ',center)
print('obstacles : ' ,polygon_obstacle_list)
print('Map boundary :',map_boundary)
