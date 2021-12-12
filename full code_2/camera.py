from math import pi
import numpy as np 
from cv2 import cv2 as cv
from matplotlib import pyplot as plt
from shapely.geometry import MultiPolygon, Polygon 
from shapely.geometry import LineString, Point


def visible(a,b, polygon_obstacle, polygon_map):
    visible = 1
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
    dist = np.linalg.norm(np.array(a)-np.array(b))
    return(dist)    

def polygon(c):
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.005 * peri, True)
    return approx

# Returns dictionnary containing for the key the index of the vertex of interest 
# and for the value, other vertices with distance that are valid paths for the robot
def vis_graph(start,targets,obstacles,polygon_obstacle,polygon_map):
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
    if color == "white":
        lower_color = np.array([0,0,128]) #np.array([125,110,90])
        upper_color = np.array([179,46,247]) #np.array([230,220,220])
    elif color == "blue":
        lower_color = np.array([96,63,62]) #np.array([60,30,20])#np.array([50,30,20])
        upper_color = np.array([130,255,255])#np.array([115,75,80])#np.array([90,55,50])
    elif color == "red":
        lower_color = np.array([130,100,100]) #np.array([70,40,150])
        upper_color = np.array([179,255,255])#np.array([130,115,200])
    elif color == "green":
        lower_color = np.array([45,60,0])#np.array([50,80,35])
        upper_color = np.array([90,140,140])#np.array([150,155,100])
    else:
        return [0]
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask_white = cv.inRange(frame, lower_color, upper_color)
    contours, _= cv.findContours(mask_white, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours


def draw_analyze_frame(frame, show_options, dilated_obstacle_list, dilated_map, visibility_graph, vertices, kalman_state, optimal_path_px, progress_idx):
    blurred_frame = cv.GaussianBlur(frame, (5,5), cv.BORDER_DEFAULT)

    #show_option 0: show contours
    #show_option 1: show estimated contours
    _, map_contour = cam_get_bounded_frame(blurred_frame, show_options[0], show_options[1])
    thymio_pos, thymio_angle, thymio_visible, thymio_radius, _ = cam_locate_thymio(blurred_frame, show_options[0], show_options[1])
    thymio_state = [thymio_pos[0], thymio_pos[1], thymio_angle]

    cam_get_targets(blurred_frame, show_options[0], show_options[1])
    cam_get_obstacles(blurred_frame, thymio_radius, show_options[0], show_options[1])

    #show_option 2: show dilated polygones
    if show_options[2]:
        for dilated_obstacle in dilated_obstacle_list:
            x_obst, y_obst = dilated_obstacle.exterior.xy
            dilated_obst_poly = list((int(point[0]),int(point[1])) for point in list(zip(x_obst, y_obst)))
            draw_polygone(blurred_frame, dilated_obst_poly, "green")

        x_map, y_map = dilated_map.exterior.xy
        dilated_map_poly = list((int(point[0]),int(point[1])) for point in list(zip(x_map, y_map)))
        draw_polygone(blurred_frame, dilated_map_poly, "white")

    #show_option 3: show visibility graph
    if show_options[3]:
        for i in visibility_graph:
            for j in visibility_graph[i]:
                cv.line(blurred_frame, [int(vertices[i][0]), int(vertices[i][1])] , [int(vertices[j[0]][0]), int(vertices[j[0]][1])], (100,100,100), 2)
                #print(vertices[i], vertices[j[0]])

    #show_option 5: show optimal path, and the progress on it
    if show_options[5]:
        draw_path(blurred_frame, optimal_path_px, progress_idx)

    #show_option 4: show kalman estimated position (drawn last to be on top)
    if show_options[4]:
        kalman_pos = (kalman_state[0], kalman_state[1])
        cv.circle(blurred_frame, kalman_pos, 7, (255, 0, 255), -1)
        cv.line(blurred_frame, kalman_pos, [int(kalman_pos[0] + 30*np.cos(kalman_state[2])), int(kalman_pos[1] - 30*np.sin(kalman_state[2]))], (255, 0, 255), 2)


def draw_polygone(frame, polygone_points, color):
    if color == "white":
        color_RGB = (255,255,255)
    elif color == "green":
        color_RGB = (0,255,0)
    elif color == "blue":
        color_RGB = (255,0,0)

    for i in range(len(polygone_points)-1):
        cv.line(frame, polygone_points[i], polygone_points[i+1], color_RGB, 3)
    cv.line(frame, polygone_points[i+1], polygone_points[0], color_RGB, 3)


def draw_path(frame, path, progress_idx):
    color_to_travel = (0,255,255) #yellow
    color_traveled = (255,255,0) #cyan

    for i in range(progress_idx):
        cv.line(frame, path[i], path[i+1], color_traveled, 3)

    for i in range(progress_idx,len(path)-1):
        cv.line(frame, path[i], path[i+1], color_to_travel, 3)

def cam_get_bounded_frame(frame, show_contour = False, show_polygone = False):
    contours_white = get_color_contour(frame, "white")

    if contours_white: #if list is not empty
        contour = max(contours_white, key = cv.contourArea)
        #print(contour)
    else:
        contour = [[0,0],[1,0]]
    x_rect,y_rect,w_rect,h_rect = cv.boundingRect(contour)

    frame_limits = [y_rect, y_rect+h_rect, x_rect, x_rect+w_rect] #min_y, max_y, min_x, max_x

    contour = contour-[x_rect,y_rect]

    if show_contour:
        cv.drawContours(frame[frame_limits[0]: frame_limits[1], frame_limits[2]: frame_limits[3]], contour, -1, (255,255,255),3)
    
    if show_polygone:
        white_pts = np.squeeze(polygon(contour))
        draw_polygone(frame, white_pts, "white")

    return frame_limits, contour


def cam_locate_thymio(frame, show_contour = False, show_circle = False):
    contours_blue = get_color_contour(frame, "blue")

    max_blue_area = 0
    thymio_pos = [0,0]
    thymio_angle = 0.0
    scale_mm = 1
    radius = 1
    blue_zones_counter = 0
    for contour_blue in contours_blue:
        area = cv.contourArea(contour_blue)
        if area > 1000 and area > max_blue_area:
            blue_zones_counter += 1
            max_blue_area = area
            (x,y),radius = cv.minEnclosingCircle(contour_blue)
            (rect_cent, (rect_width, rect_height), rect_angle) = cv.minAreaRect(contour_blue)
            center = (int(x),int(y))
            scale_mm = 140/(2*radius)
            radius = int(radius)
            thymio_pos = center
            if show_contour:
                cv.drawContours(frame, contour_blue, -1, (255,0,0),3)
            if show_circle:
                cv.circle(frame,center,radius,(255,0,0),2)
                cv.circle(frame, center, 7, (255, 0, 0), -1)
                box = cv.boxPoints((rect_cent, (rect_width, rect_height), rect_angle))
                box_points = list((int(point[0]),int(point[1])) for point in box)
                draw_polygone(frame,box_points,"blue")
            #takes the angle from the rectangle framing the thymio that is the closest to the noisy angle
            noisy_angle = np.arctan2(center[1] - rect_cent[1],rect_cent[0] - center[0]) #is in rad
            possible_angles = np.array(list((-rect_angle + s*90.)*pi/180. for s in range(4)))
            angles_errors = list(abs(np.arctan2(np.sin(noisy_angle - poss_angle), np.cos(noisy_angle - poss_angle))) for poss_angle in possible_angles)
            thymio_angle = possible_angles[np.argmin(angles_errors)]
    thymio_visible = True

    if blue_zones_counter == 0:
        thymio_visible = False

    return thymio_pos, thymio_angle, thymio_visible, radius, scale_mm


def cam_get_contour(map_contour, radius):
    map_contour_poly = Polygon(np.squeeze(polygon(map_contour)))
    dilated_map = map_contour_poly.buffer(-radius, join_style=2,single_sided=True)
    return map_contour_poly, dilated_map


def cam_get_targets(frame, show_contour = False, show_center = False):
    contours_red = get_color_contour(frame, "red")

    target_list = []
    for contour_red in contours_red:
        area = cv.contourArea(contour_red)
        if area > 1000:
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



def cam_get_obstacles(frame, radius, show_contour = False, show_polygone = False):
    contours_green = get_color_contour(frame, "green")

    area_obstacles = MultiPolygon()
    dilated_obstacle_list = []
    for contour_green in contours_green:
        area = cv.contourArea(contour_green)
        if area > 1000:
            pts = np.squeeze(polygon(contour_green))
            pol = Polygon(pts)
            dilated_obstacle = pol.buffer(radius, join_style=2 ,single_sided=True)
            dilated_obstacle_list.append(dilated_obstacle)
            area_obstacles = area_obstacles.union(dilated_obstacle)
            if show_contour:
                cv.drawContours(frame, contour_green, -1, (0,255,0),3)
            if show_polygone:
                draw_polygone(frame, pts, "green")

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
    return [coords_px[0]*scale, (y_axis_size-coords_px[1])*scale]

def convert_to_px(y_axis_size, scale, coords_mm):
    return [int(coords_mm[0]/scale), int(y_axis_size-coords_mm[1]/scale)]

def reachable_targets(targets_list,dilated_map,dilated_obstacle_list):
    for target in targets_list:
        if not Point(target).within(dilated_map):
            targets_list.remove(target)
        else:
          for x in dilated_obstacle_list:
             if Point(target).within(x):
                 targets_list.remove(target)

    return targets_list

def object_detection(frame):
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
    map_contour_polygone, dilated_map = cam_get_contour(map_contour, thymio_radius)
    #locate the targets
    targets_list = cam_get_targets(bounded_frame)
    #locate the obstacles
    obstacles_list, dilated_obstacle_list  = cam_get_obstacles(bounded_frame, thymio_radius)
    #select only reachable targets
    targets_list = reachable_targets(targets_list,dilated_map,dilated_obstacle_list)


    return thymio_state,targets_list,obstacles_list,dilated_obstacle_list,dilated_map, frame_limits, scale_mm