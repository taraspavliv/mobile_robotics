"""
camera.py

This file contains all the function that are useful for the computer vision going from detecting the map, obstacles,
targets, and the robot to computing the visibility graph which can be used to determine the shortest path to visit all 
the targets without collision. It also provides some useful visualization functions"
"""

import numpy as np 
from cv2 import cv2 as cv
from shapely.geometry import MultiPolygon, Polygon, LineString, Point


MIN_AREA = 1000
THYMIO_DIAM = 140 #mm
POLYGON_THRESHOLD = 0.005

#Returns 
def visible(a,b, polygon_obstacle, polygon_map):
    """visible function determines if the robot can travel
       between two points using a straigth line without leaving 
       the map or touch an obstacle
    :param a: origin point
    :param b: destination point
    :param polygon_obstacle: list of polygon representing the dilated obstacles
    :param polygon_map: polygon representing the dilated map
    :return visible: True if the line does not cross obstacles or leave map
                     False otherwise
    """
    visible = True
    line = LineString([a,b])
    for x in polygon_obstacle:
        within_obstacle = line.within(x)
        crosses_obstacle = line.crosses(x)
        if within_obstacle == True or crosses_obstacle == True:
            visible = False     
    in_map = polygon_map.contains(line)
    if in_map == False:
        visible = False
    return visible

def distance(a,b):
    """distance function compute the eucledean distance between two points
    :param a: first point
    :param b: second point
    :return dist: eucledean distance between points a and b
    """
    dist = np.linalg.norm(np.array(a)-np.array(b))
    return(dist)    

def polygon(c):
    """polygon function computes the polygon approximation of a curve
    :param c: curve that we want to approximate
    :return approx: approximation of the curve
    """
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, POLYGON_THRESHOLD * peri, True)
    return approx


def vis_graph(start,targets,obstacles,polygon_obstacle,polygon_map):
    """vis_graph function computes the visibility graph based on the computer vision results
     :param start: position of the robot
     :param targets: list of positions of the targets
     :param obstacles: list of vertices of the obstacles
     :param polygon_obstacle: list of polygons of dilated obstacles
     :param polygon_map: polygon of the dilated map
     :return graph: dictionnary containing in the key the index of the vertex of interest 
                     and for the value, other vertices with distance that are valid paths
                     for the robot
     :return start_idx: index of the robot vertex in vertices list
     :return targets_idx_list: list of the indexes of the targets (based on vertices list)
     :return vertices: List of all vertices, robot position, vertices of obstacles, and targets
     """
    graph = {}
    start_idx = 0
    obstacles = [item for sublist in obstacles for item in sublist]
    targets_idx_list = list(range(1,len(targets)+1))
    vertices_with_rep = [start] + targets + obstacles
    vertices = []
    for i in vertices_with_rep : 
        if i not in vertices: 
           vertices.append(i)

    for vtx1_idx, vtx1 in enumerate(vertices):
        for vtx2_idx, vtx2 in enumerate(vertices):
            if vtx1_idx != vtx2_idx:
                if visible(vtx1,vtx2,polygon_obstacle,polygon_map)==1:
                    if vtx1_idx in graph :
                        graph[vtx1_idx].append([vtx2_idx,distance(vtx1,vtx2)])
                    else:
                        graph[vtx1_idx] = [(vtx2_idx,distance(vtx1,vtx2))]
                else:
                    if not(vtx1_idx in graph):
                        graph[vtx1_idx] = []

    return graph, start_idx, targets_idx_list, vertices

def get_color_contour(frame, color):
    """get_color_contour funtion convert the frame to HSV colour space
       then, apply a mask for the color chosen and returns the contours
       of this mask.
    :param frame: frame of a video
    :param color: color to be detected by the ranges
    :return contours: contours of the mask of the colour detected
    """
    if color == "white":
        lower_color = np.array([0,0,128]) 
        upper_color = np.array([179,46,247])
    elif color == "blue":
        lower_color = np.array([96,63,62])
        upper_color = np.array([130,255,255])
    elif color == "red":
        lower_color = np.array([130,100,100])
        upper_color = np.array([179,255,255])
    elif color == "green":
        lower_color = np.array([45,60,0])
        upper_color = np.array([90,140,140])
    else:
        return [0]
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask_white = cv.inRange(frame, lower_color, upper_color)
    contours, _= cv.findContours(mask_white, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours


def draw_analyze_frame(frame, show_options, dilated_obstacle_list, dilated_map, visibility_graph, vertices, kalman_state, optimal_path_px, progress_idx):
    """draw_analyze_frame helps to visualize the processing of the frame
    :params frame: frame on which the computer vision is done
    :params show_options: different drawing options
    :params dilated_obstacle_list: list of the dilated obstacles
    :params dilated_map: the dilated map
    :params visibility_graph: dictionnary containing the visibility graph
    :params vertices: all vertices (robot position, targets, vertices of dilated polygon obstacles)
    :params kalman_state: pose of the robot (position and angle) estimated by the kalman filter
    :params optimal_path_px: vertices of the shortest path to visit all the targets
    :params progress_idx: vertex index of the path reached by the robot
    :return blurred_frame: frame on which we applied a gaussian blur to cancel noise
    :return thymio_state: robot position and angle
    :return thymio_visible: boolean set to true if the camera detects the robot
    """
    blurred_frame = cv.GaussianBlur(frame, (5,5), cv.BORDER_DEFAULT)

    #show_option 0: show contours
    #show_option 1: show estimated contours
    _, map_contour = cam_get_bounded_frame(blurred_frame, show_options[0], show_options[1])
    thymio_pos, thymio_angle, thymio_visible, thymio_radius, _ = cam_locate_thymio(blurred_frame, show_options[0], show_options[1])
    thymio_state = [thymio_pos[0], thymio_pos[1], thymio_angle]

    #doesn't need to return, we draw in the functions
    cam_get_targets(blurred_frame, show_options[0], show_options[1])
    cam_get_obstacles(blurred_frame, thymio_radius, show_options[0], show_options[1])

    #show_option 2: show dilated polygons
    if show_options[2]:
        for dilated_obstacle in dilated_obstacle_list:
            x_obst, y_obst = dilated_obstacle.exterior.xy
            dilated_obst_poly = list((int(point[0]),int(point[1])) for point in list(zip(x_obst, y_obst)))
            draw_polygon(blurred_frame, dilated_obst_poly, "green")

        x_map, y_map = dilated_map.exterior.xy
        dilated_map_poly = list((int(point[0]),int(point[1])) for point in list(zip(x_map, y_map)))
        draw_polygon(blurred_frame, dilated_map_poly, "white")

    #show_option 3: show visibility graph
    if show_options[3]:
        for i in visibility_graph:
            for j in visibility_graph[i]:
                cv.line(blurred_frame, [int(vertices[i][0]), int(vertices[i][1])] , [int(vertices[j[0]][0]), int(vertices[j[0]][1])], (100,100,100), 2)

    #show_option 5: show optimal path, and the progress on it
    if show_options[5]:
        draw_path(blurred_frame, optimal_path_px, progress_idx)

    #show_option 4: show kalman estimated position (drawn last to be on top)
    if show_options[4]:
        kalman_pos = (kalman_state[0], kalman_state[1])
        #draws center
        cv.circle(blurred_frame, kalman_pos, 7, (255, 0, 255), -1)
        #draws direction
        cv.line(blurred_frame, kalman_pos, [int(kalman_pos[0] + 30*np.cos(kalman_state[2])), \
                                            int(kalman_pos[1] - 30*np.sin(kalman_state[2]))], (255, 0, 255), 2)

    return blurred_frame, thymio_state, thymio_visible


def draw_polygon(frame, polygon_points, color):
    """draw_polygon function draws polygon on the frame
    :params frame: frame on which we do the computer vision tasks
    :params polygon_points: vertices of the polygon
    :params color: color chosen for plotting the polygon
    """
    if color == "white":
        color_RGB = (255,255,255)
    elif color == "green":
        color_RGB = (0,255,0)
    elif color == "blue":
        color_RGB = (255,0,0)

    for i in range(len(polygon_points)-1):
        cv.line(frame, polygon_points[i], polygon_points[i+1], color_RGB, 3)
    cv.line(frame, polygon_points[i+1], polygon_points[0], color_RGB, 3)


def draw_path(frame, path, progress_idx):
    """draw_path function draws the shortest path, using one colour for the 
       part of the path traveled and another one for the rest of the path
    :params frame: frame on which we do the computer vision tasks
    :params path: vertices of the shortest path
    :params progress_idx: vertex index of the path reached by the robot
    """
    color_to_travel = (0,255,255) #yellow
    color_traveled = (255,255,0) #cyan

    for i in range(progress_idx): #traveled path
        cv.line(frame, path[i], path[i+1], color_traveled, 3)

    for i in range(progress_idx,len(path)-1): #path left to travel
        cv.line(frame, path[i], path[i+1], color_to_travel, 3)

def cam_get_bounded_frame(frame, show_contour = False, show_polygon = False):
    """cam_get_bounded_frame function finds the contour of the map and compute the frame 
       limits in order to crop it later.
    :params frame: frame on which we do the computer vision tasks
    :params show_contour: drawing option for displaying the contours
    :params show_polygon: drawing option for displaying the polygon
    :return frame_limits: rectangle bounding the map
    :return contour: contour of the map
    """
    contours_white = get_color_contour(frame, "white")

    contour = max(contours_white, key = cv.contourArea)
    x_rect,y_rect,w_rect,h_rect = cv.boundingRect(contour)

    frame_limits = [y_rect, y_rect+h_rect, x_rect, x_rect+w_rect]  #min_y, max_y, min_x, max_x coordinates

    contour = contour-[x_rect,y_rect]

    if show_contour:
        cv.drawContours(frame[frame_limits[0]: frame_limits[1], frame_limits[2]: frame_limits[3]], contour, -1, (255,255,255),3)
    
    if show_polygon:
        white_pts = np.squeeze(polygon(contour))
        draw_polygon(frame, white_pts, "white")

    return frame_limits, contour


def cam_locate_thymio(frame, show_contour = False, show_circle = False):
    """cam_locate_thymio function detects the thymio position and angle
    :params frame: frame on which we do the computer vision tasks
    :params show_contour: option for displaying the contour of the thymio
    :params show_circle: option for displaying the minimum enclosing circle of the contour
    :return thymio_pos: position of the center of the minimum enclosing circle
    :return thymio_angle: angle of the thymio based on the minimum area rectangle enclosing the contour
    :return thymio_visible: Boolean set to true if the camera detects the robot
    :return radius: radius of the minimum enclosing circle for the robot's contour
    :return scale_mm: scale for conversion to pixels to mm [mm/pixel]
    """
    contours_blue = get_color_contour(frame, "blue")

    max_blue_area = 0
    thymio_pos = [0,0]
    thymio_angle = 0.0
    scale_mm = 1
    radius = 1
    blue_zones_counter = 0
    for contour_blue in contours_blue:
        area = cv.contourArea(contour_blue)
        if area > MIN_AREA and area > max_blue_area:
            blue_zones_counter += 1
            max_blue_area = area
            (x,y),radius = cv.minEnclosingCircle(contour_blue)
            (rect_cent, (rect_width, rect_height), rect_angle) = cv.minAreaRect(contour_blue)
            center = (int(x),int(y))
            scale_mm = THYMIO_DIAM/(2*radius)
            radius = int(radius)
            thymio_pos = center
            if show_contour:
                cv.drawContours(frame, contour_blue, -1, (255,0,0),3)
            if show_circle:
                cv.circle(frame,center,radius,(255,0,0),2)
                cv.circle(frame, center, 7, (255, 0, 0), -1)
                box = cv.boxPoints((rect_cent, (rect_width, rect_height), rect_angle))
                box_points = list((int(point[0]),int(point[1])) for point in box)
                draw_polygon(frame,box_points,"blue")
            #takes the angle from the rectangle framing the thymio that is the closest to the noisy angle
            noisy_angle = np.arctan2(center[1] - rect_cent[1],rect_cent[0] - center[0]) #is in rad
            possible_angles = np.array(list((-rect_angle + s*90.)*np.pi/180. for s in range(4)))
            angles_errors = list(abs(np.arctan2(np.sin(noisy_angle - poss_angle), np.cos(noisy_angle - poss_angle))) for poss_angle in possible_angles)
            thymio_angle = possible_angles[np.argmin(angles_errors)]
    thymio_visible = True

    if blue_zones_counter == 0:
        thymio_visible = False

    return thymio_pos, thymio_angle, thymio_visible, radius, scale_mm


def cam_get_contour(map_contour, radius):
    """cam_get_contour function approximates the map contour as a polygon and dilates it towards its interior by the robot's radius
    :params map_contour: contour of the map
    :params radius: radius of the robot in pixels
    :return map_contour_poly: polygon approximating the map contour
    :return dilated_map: dilated map_contour_poly by the robots radius towards its interior
    """
    map_contour_poly = Polygon(np.squeeze(polygon(map_contour)))
    dilated_map = map_contour_poly.buffer(-radius, join_style=3 ,single_sided=True)
    return map_contour_poly, dilated_map


def cam_get_targets(frame, show_contour = False, show_center = False):
    """cam_get_targets function detects the targets, and stores their centroid
    :params frame: frame on which we do the computer vision tasks
    :params show_contour: drawing option for the contour
    :params show_center: drawing option of the centroid of the targets
    :return target_list: list of the centroids coordinates of the targets
    """
    contours_red = get_color_contour(frame, "red")

    target_list = []
    for contour_red in contours_red:
        area = cv.contourArea(contour_red)
        if area > MIN_AREA:
            M = cv.moments(contour_red)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                target_list.append((cx,cy))
            if show_contour:
                cv.drawContours(frame, contour_red, -1, (0,0,255),3)
            if show_center:
                cv.circle(frame, (cx, cy), 7, (0,0,255), -1)

    return target_list



def cam_get_obstacles(frame, radius, show_contour = False, show_polygon = False):
    """cam_get_obstacles function detects obstacles and dilate them by the robot's radius
    :params frame: frame on which we do the computer vision tasks
    :params radius: radius of the robot in pixels
    :params show_contour: drawing option for contours
    :params show_polygon: drawing option for polygons
    :return obstacles_list: list of vertices of the obstacles
    :return dilated_obstacle_list: list of polygons of dilated obstacles
    """
    contours_green = get_color_contour(frame, "green")

    area_obstacles = MultiPolygon()
    dilated_obstacle_list = []
    for contour_green in contours_green:
        area = cv.contourArea(contour_green)
        if area > MIN_AREA:
            pts = np.squeeze(polygon(contour_green))
            pol = Polygon(pts)
            dilated_obstacle = pol.buffer(radius, join_style=3 ,single_sided=True)
            dilated_obstacle_list.append(dilated_obstacle)
            area_obstacles = area_obstacles.union(dilated_obstacle)
            if show_contour:
                cv.drawContours(frame, contour_green, -1, (0,255,0),3)
            if show_polygon:
                draw_polygon(frame, pts, "green")

    obstacles_boundary = []

    if type(area_obstacles) != Polygon:
        for geom in area_obstacles.geoms:    
            xo, yo = geom.exterior.xy
            obstacles_boundary.append(zip(xo,yo))

    obstacles_list = []
    for obstacle in obstacles_boundary :
        obstacles_list.append(list(obstacle))

    return obstacles_list, dilated_obstacle_list


def convert_to_mm(y_axis_size, scale, coords_px):
    """convert_to_mm function converts pixels to mm and invert y axis
    :params y_axis_size: width of the frame 
    :params scale: scale [mm/pixel]
    :params coords_px: coordinates in pixels
    :return: coordinates in mm
    """
    return [coords_px[0]*scale, (y_axis_size-coords_px[1])*scale]

def convert_to_px(y_axis_size, scale, coords_mm):
    """convert_to_px function convert mm to pixels and invert y axis
    :params y_axis_size: width of the frame 
    :params scale: scale [mm/pixel]
    :params coords_mm: coordinates in mm
    :return: coordinates in pixels
    """
    return [int(coords_mm[0]/scale), int(y_axis_size-coords_mm[1]/scale)]

def reachable_targets(targets_list,dilated_map,dilated_obstacle_list):
    """reachable_targets function gets the targets that are in the dilated map and
       outside of the dilated obstacles
    :params targets_list: list of all the detected targets
    :params dilated_map: polygon of the dilated map
    :params dilated_obstacle_list: list of dilated polygons
    :return targets_list: targets list updated with only reachable targets
    """
    for target in targets_list:
        if not Point(target).within(dilated_map):
            targets_list.remove(target)
        else:
          for x in dilated_obstacle_list:
             if Point(target).within(x):
                 targets_list.remove(target)

    return targets_list

def object_detection(frame):
    """object_detection_function do all the computer vision tasks using the other functions
    :params frame: frame on which we do the computer vision tasks
    :return thymio_state: robot position and angle
    :return targets_list : list of the centroids of the reachable targets
    :return obstacles_list : list of the obstacles
    :return dilated_obstacle_list : list of the obstacles dilated by the robot's radius
    :return dilated_map : map dilated by the robot's radius
    :return frame_limits : coordinates of the cropped frame based on the rectangle bounding the map
    :return scale_mm : scale to convert from pixels to mm [mm/pixel]
    """
    #blur image to have less noise
    blurred_frame = cv.GaussianBlur(frame, (5,5), cv.BORDER_DEFAULT)
    #find the map boundaries
    frame_limits, map_contour = cam_get_bounded_frame(blurred_frame)
    #reduce the working frame to the boundaries
    bounded_frame = blurred_frame[frame_limits[0]: frame_limits[1], frame_limits[2]: frame_limits[3]]
    #locate the robot
    thymio_pos, thymio_angle, thymio_visible, thymio_radius, scale_mm = cam_locate_thymio(bounded_frame)
    thymio_state = [thymio_pos[0], thymio_pos[1], thymio_angle]
    #reduce the boundaries so the whole robot stays inside when the center is on the "dilated_map"
    map_contour_polygon, dilated_map = cam_get_contour(map_contour, thymio_radius)
    #locate the targets
    targets_list = cam_get_targets(bounded_frame)
    #locate the obstacles
    obstacles_list, dilated_obstacle_list  = cam_get_obstacles(bounded_frame, thymio_radius)
    #select only reachable targets
    targets_list = reachable_targets(targets_list,dilated_map,dilated_obstacle_list)

    return thymio_state,targets_list,obstacles_list,dilated_obstacle_list,dilated_map, frame_limits, scale_mm