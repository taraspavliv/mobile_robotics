import threading
import time
import asyncio

from camera import *
from path_planning import *
from kalman import *
from navigation import *

#For the camera
capture = cv.VideoCapture(1 + cv.CAP_DSHOW)
capture.set(cv.CAP_PROP_BUFFERSIZE, 1)

## Global variables because shared between threads

#For navigation
optimal_path = []
convert_px_mm = 1
objectif_number = 0

#For display
frame_limits = []
visibility_graph = []
vertices = []
dilated_obstacle_list = []
dilated_map = []

#For kalman
mu = np.array([0.,0.,0.,0.,0.,0.]) #x,y,x_speed,y_speed,angle, angular_speed
motor_cmd = np.array([0,0]) #motor command
thymio_cam_state = [0,0,0] #pos_x, pos_y, angle
thymio_visible = False

#For all threads, when one is stopped, the other ones stop too
stop_threads = False

def setup():
    global capture, convert_px_mm, optimal_path, frame_limits
    global visibility_graph, vertices, dilated_obstacle_list, dilated_map, mu

    #we skip a few frames between taking the one to initialize the map
    valid_image = False
    frame_counter = 0
    while frame_counter < 20:
        valid_image, first_image = capture.read()
        if valid_image == True:
            frame_counter += 1
    
    #We detect all objects and the dilated versions for the visibility graph
    thymio_state,targets_list,obstacles_list,dilated_obstacle_list,dilated_map, frame_limits, convert_px_mm = object_detection(first_image)

    #For the kalman filter, we iniatialize mu (we also convert from pixels to millimeters)
    mu_pos_estim = convert_to_mm(frame_limits[1] - frame_limits[0], convert_px_mm, [thymio_state[0],thymio_state[1]]) #converts to mm
    mu = np.array([mu_pos_estim[0],mu_pos_estim[1],0.,0.,thymio_state[2],0.])
    
    #We build the visibility graph, and save the indices of the thymio and the targets
    visibility_graph,start_idx,targets_idx_list,vertices = vis_graph([thymio_state[0], thymio_state[1]],targets_list,obstacles_list, \
                                                                      dilated_obstacle_list,dilated_map)

    #Create an array of distance between all pairs of points: targets and the thymio start position
    distance_array, path_array = create_distance_path_matrix(visibility_graph,start_idx, targets_idx_list)

    #Calculate the sortest path to go to all targets from start positon
    total_distance, targets_idx_order = shortest_path(start_idx, np.array([start_idx]), distance_array)

    #Build the optimal path coordinates (in pixels)
    optimal_path_px = reconstruct_optimal_path(path_array, targets_idx_order, vertices)

    #Convert the optimal path from coordinates in pixels to coordinates in millimeters
    optimal_path_mm = []
    for point in optimal_path_px:
        point_in_mm = convert_to_mm(frame_limits[1] - frame_limits[0], convert_px_mm, [point[0],point[1]])
        optimal_path_mm.append(point_in_mm)
    optimal_path = np.array(optimal_path_mm)

setup()

#Responsible to find the thymio on the camera and to display important information on a video
def cam_thread():
    global capture, convert_px_mm, thymio_cam_state, thymio_visible, frame_limits, mu, objectif_number
    global optimal_path, visibility_graph, vertices, dilated_obstacle_list, dilated_map, stop_threads

    #rescale the video
    img_scale = 0.75

    #Options on what to display
    show_contours = False
    show_polygones = False
    show_dilated_polygones = False
    show_visibility_graph = False
    show_optimal_path = False
    show_kalman_estimation = False
    show_option = [show_contours, show_polygones, show_dilated_polygones, show_visibility_graph, show_kalman_estimation, show_optimal_path]
    optimal_path_px = list([int(convert_to_px(frame_limits[1] - frame_limits[0], convert_px_mm, s)[0]), \
                            int(convert_to_px(frame_limits[1] - frame_limits[0], convert_px_mm, s)[1])] for s in optimal_path)
    while True:
        #Read the image
        valid_image, frame = capture.read()
        if stop_threads or not valid_image:
            stop_threads = True
            break

        #Cut borders of the video to display only the terrain
        bounded_frame = frame[frame_limits[0]: frame_limits[1], frame_limits[2]: frame_limits[3]]

        #Transform the esimated kalman position to pixel values (for display)
        kalman_est_pos = convert_to_px(frame_limits[1] - frame_limits[0], convert_px_mm, [mu[0], mu[1]]) #converts kalman estimated pos to pixel pos
        kalman_est = [kalman_est_pos[0], kalman_est_pos[1], mu[4]]

        #Analyze the video image to find the thymio (if he's visible) and to draw some information depending on the options 
        modified_frame, thymio_state, thymio_visible = draw_analyze_frame(bounded_frame, show_option, dilated_obstacle_list, dilated_map,\
                                                                          visibility_graph, vertices, kalman_est, optimal_path_px, objectif_number)

        #Convert the thymio found on the video to position in millimeter (for the kalman filter)
        thymio_pos_estim = convert_to_mm(frame_limits[1] - frame_limits[0], convert_px_mm, [thymio_state[0],thymio_state[1]])
        thymio_cam_state = [thymio_pos_estim[0], thymio_pos_estim[1] ,thymio_state[2]] 
    
        #Rescale the video
        dim = (int(modified_frame.shape[1]*img_scale), int(modified_frame.shape[0]*img_scale))
        cv.imshow('Video', cv.resize(modified_frame, dim))

        #Keys to toggle shown information options
        key_pressed = cv.waitKey(20) #time discretization: 50Hz
        if key_pressed == ord('q'):
            show_contours = not show_contours
        if key_pressed == ord('w'):
            show_polygones = not show_polygones
        if key_pressed == ord('e'):
            show_dilated_polygones = not show_dilated_polygones
        if key_pressed == ord('r'):
            show_visibility_graph = not show_visibility_graph
        if key_pressed == ord('t'):
            show_kalman_estimation = not show_kalman_estimation
        if key_pressed == ord('z'):
            show_optimal_path = not show_optimal_path
        if key_pressed == ord('d'):
            stop_threads = True
            break
        show_option = [show_contours, show_polygones, show_dilated_polygones, show_visibility_graph, show_kalman_estimation, show_optimal_path]

threading.Thread(target=cam_thread).start()

def kalman_thread():
    global mu, thymio_cam_state, motor_cmd, thymio_visible, stop_threads

    #Convert speed commands in mm/s
    speed_conv = 0.36
    
    #Variance initialization
    sig_init = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
    T1 = 0.03 #time discretization: 33Hz
    r = 47 # half of the distance between the wheel

    #Uncertainty on state
    R = np.array([[0.01,0.,0.,0.,0.,0.], \
                  [0.,0.01,0.,0.,0.,0.], \
                  [0.,0.,0.01,0.,0.,0.], \
                  [0.,0.,0.,0.01,0.,0.], \
                  [0.,0.,0.,0.,0.00001,0.], \
                  [0.,0.,0.,0.,0.,0.1]])

    #Uncertainty on measurement
    Q = np.array([[0.01,0.,0.],[0.,0.01,0.],[0.,0.,0.01]])

    mu_prev = mu
    sig_prev = sig_init
    
    startKal = 0
    endKal = T1
    while True:
        u = speed_conv*np.array([motor_cmd[1], motor_cmd[0]])
        meas = thymio_cam_state
        #We use of a counter to keep track of time passed, for an accurate T1
        endKal = time.perf_counter()
        T1 = (endKal - startKal)
        (mu,sig) = kalmanFilter(mu_prev, sig_prev, u, meas, T1, r, R, Q, thymio_visible)
        startKal = time.perf_counter()
        mu_prev = np.concatenate(np.transpose(mu))
        sig_prev = sig
        time.sleep(0.03)
        if stop_threads:
            break

threading.Thread(target=kalman_thread).start()


async def navigation_thread():
    global motor_cmd, mu, optimal_path, objectif_number
    #Initialisation step
    prev_error = 0
    T1 = 0.01 #time discretization: 100Hz
    node = await client.wait_for_node()

    startKal = 0
    endKal = T1
    turn = True #state of the robot: starts by turning to the right direction
    while True:
        await node.lock()
        pos_r = np.array([mu[0][0], mu[1][0]])
        angle_r = mu[4]

        #we use of a counter to keep track of time passed, for an accurate T1
        endKal = time.perf_counter() 
        T1 = (endKal - startKal)  
        motor_cmd, prev_error, objectif_number, turn = navigation(pos_r, angle_r, optimal_path, prev_error, T1, objectif_number, turn )
        startKal = time.perf_counter()
        node.send_set_variables(motors_command(motor_cmd))

        #Gets the motors commands from the Thymio in case of there is an unforseen obstacle to use it in the Kalman
        async def get_mot_comm(s):   
            await node.wait_for_variables({str(s)})
            return node[s]
        motor_left=await get_mot_comm("motor.left.target")
        motor_right=await get_mot_comm("motor.right.target")
        motor_cmd=np.array([motor_left, motor_right])

        time.sleep(0.03)
        if stop_threads:
            node.send_set_variables(motors_command(np.array([0,0])))
            await node.unlock()
            break
        await node.unlock()

threading.Thread(target=asyncio.run(navigation_thread())).start()