import threading
import time
import asyncio

from camera import *
from path_planning import *
from kalman import *
from navigation import *
optimal_path = []


capture = cv.VideoCapture('new.mp4')

stop_threads = False

#for navigation
convert_px_mm = 1

#for display
frame_limits = []
visibility_graph = []
vertices = []
dilated_obstacle_list = []
dilated_map = []

#for kalman
mu = np.array([0.,0.,0.,0.,0.,0.]) #x,y,x_speed,y_speed,angle, angular_speed
motor_cmd = np.array([0,0]) #motor command
thymio_cam_state = [0,0,0] #ideally pos_x, pos_y, angle
thymio_visible = False

def setup():
    global capture, convert_px_mm, optimal_path, frame_limits
    global visibility_graph, vertices, dilated_obstacle_list, dilated_map, mu

    #we will analyze the first valid image
    valid_image = False
    while not valid_image:
        valid_image, first_image = capture.read()
    
    #we detect all objects and the dilated versions for the visibility graph
    thymio_state,targets_list,obstacles_list,dilated_obstacle_list,dilated_map, frame_limits, convert_px_mm = object_detection(first_image)

    #for kalman filter, we iniate the mu
    mu_pos_estim = convert_to_mm(frame_limits[1] - frame_limits[0], convert_px_mm, [thymio_state[0],thymio_state[1]]) #converts to mm
    #mu = np.array([mu_pos_estim[0],mu_pos_estim[1],0.,0.,thymio_state[2],0.])

    #we build the visibility graph
    visibility_graph,start_idx,targets_idx_list,vertices = vis_graph([thymio_state[0], thymio_state[1]],targets_list,obstacles_list,dilated_obstacle_list,dilated_map)

    #create an array of distance between all pairs of target and the thymio start position
    distance_array, path_array = create_distance_path_matrix(visibility_graph,start_idx,[1,3])

    #calculate the sortest path to go to all targets from start positon
    total_distance, targets_idx_order = shortest_path(0, np.array([0]), distance_array)

    #save the optimal path
    for i in range(len(targets_idx_order)-1):
        for j in range(len(path_array[i][int(targets_idx_order[i+1])])):
            if i != 0 and j == 0: 
                continue
            optimal_path.append(vertices[path_array[i][int(targets_idx_order[i+1])][j]])

    #convert the optimal path (for now in pixels) to optimal path in millimeters
    optimal_path_mm = []
    for point in optimal_path:
        point_in_mm = convert_to_mm(frame_limits[1] - frame_limits[0], convert_px_mm, [point[0],point[1]]) #converts to mm
        optimal_path_mm.append(point_in_mm)
    optimal_path = np.array(optimal_path_mm)


setup()

#responsible to find the thymio on the camera and to display the important information on a video
def cam_thread():
    global capture, convert_px_mm, thymio_cam_state, thymio_visible, frame_limits, mu
    global optimal_path, visibility_graph, vertices, dilated_obstacle_list, dilated_map, stop_threads

    img_scale = 0.5
    #options on what to display
    show_contours = False
    show_polygones = False
    show_dilated_polygones = False
    show_visibility_graph = False
    show_kalman_estimation = False
    show_option = [show_contours, show_polygones, show_dilated_polygones, show_visibility_graph, show_kalman_estimation]
    while True:

        #read the image
        valid_image, frame = capture.read()
        if stop_threads or not valid_image:
            stop_threads = True
            break

        #cut borders of the video to display only the terrain
        bounded_frame = frame[frame_limits[0]: frame_limits[1], frame_limits[2]: frame_limits[3]]

        #transform the esimated kalman position to pixel values
        kalman_est_pos = convert_to_px(frame_limits[1] - frame_limits[0], convert_px_mm, [mu[0], mu[1]]) #converts kalman estimated pos to pixel pos

        #analyze the video image to find the thymio (if he's visible) and to draw some information depending on the options 
        modified_frame, thymio_state, thymio_visible = draw_analyze_frame(bounded_frame, show_option, dilated_obstacle_list, dilated_map, kalman_est_pos)

        #convert the found thymio from the video to position in millimeter
        thymio_pos_estim = convert_to_mm(frame_limits[1] - frame_limits[0], convert_px_mm, [thymio_state[0],thymio_state[1]]) #converts thymio pos to mm
        thymio_cam_state = [thymio_pos_estim[0], thymio_pos_estim[1] ,thymio_state[2]] 
    
        #rescale the video
        dim = (int(modified_frame.shape[1]*img_scale), int(modified_frame.shape[0]*img_scale))
        cv.imshow('Video', cv.resize(modified_frame, dim))

        #keys to toggle shown information options
        key_pressed = cv.waitKey(1000)
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
        if key_pressed == ord('d'):
            stop_threads = True
            break

        show_option = [show_contours, show_polygones, show_dilated_polygones, show_visibility_graph, show_kalman_estimation]

threading.Thread(target=cam_thread).start()






def kalman_thread():
    global mu, thymio_cam_state, motor_cmd, thymio_visible, stop_threads
    #Convert speed commands in mm/s
    speed_conv = 0.45
    
    #J'essaie avec des valeudrs au bol...
    sig_init = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
    T1 = 0.03 #33Hz
    r = 47 #mm
    #uncertainty on state
    R = np.array([[0.01,0.,0.,0.,0.,0.],[0.,0.01,0.,0.,0.,0.],[0.,0.,0.01,0.,0.,0.],[0.,0.,0.,0.01,0.,0.],[0.,0.,0.,0.,0.0000000001,0.],[0.,0.,0.,0.,0.,0.01]])
    #uncertainty on measurement
    Q = np.array([[0.01,0.],[0.,0.01]])

    mu_prev = mu
    sig_prev = sig_init

    while True:
        thymio_visible=False
        u = speed_conv*np.array([motor_cmd[1], motor_cmd[0]])
        meas = (thymio_cam_state[0], thymio_cam_state[1])
        (mu,sig) = kalmanFilter(mu_prev, sig_prev, u, meas, T1, r, R, Q, thymio_visible)
        mu_prev = np.concatenate(np.transpose(mu))
        sig_prev = sig
        time.sleep(T1)
        if stop_threads:
            break

threading.Thread(target=kalman_thread).start()


#Mu position mm, speed mm/s, angle rad, vitesse angulaire

async def navigation_thread():
    global motor_cmd, mu, optimal_path
    prev_error = 0
    T1 = 0.01
    objectif_number = 0
    node = await client.wait_for_node()
    #print(optimal_path)
    while True:
        #optimal_path=np.array([[0,0],[1000,0]])
        optimal_path=np.array([[0,0],[100,0],[0,0]])
        #optimal_path=np.array([[0,0],[200,0],[200,200],[0,200],[0,0]])
        #optimal_path=np.array([[0,0],[-200,0],[-200,-200],[0,-200],[0,0]])
        await node.lock()
        #pos_r, angle_r, obj_list, prev_err_pos, T, objectif_number
        pos_r = np.array([mu[0][0], mu[1][0]])
        #test#########################################################
        angle_r = mu[4]#%(2*np.pi)
        #print("ANGLE",angle_r)
        #print("POS", pos_r)
        motor_cmd, prev_error, objectif_number = navigation(pos_r, angle_r, optimal_path, prev_error, T1, objectif_number)
        node.send_set_variables(motors_command(motor_cmd))




        async def get_mot_comm(s):   
            await node.wait_for_variables({str(s)})
            return node[s]
        motor_left=await get_mot_comm("motor.left.target")
        motor_right=await get_mot_comm("motor.right.target")
        motor_cmd=np.array([motor_left, motor_right])
        print(motor_cmd)
        #print(motors)
        
        #print(pos_r)



        #print(motors_cmd)
        time.sleep(T1)
        if stop_threads:
            node.send_set_variables(motors_command(np.array([0,0])))
            await node.unlock()
            break
        await node.unlock()
threading.Thread(target=asyncio.run(navigation_thread())).start()



#Todo Retourner la speed des moteurs quand le thymio passe en navigation locale-> old_u