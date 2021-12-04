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

#for display
frame_limits = []
visibility_graph = []
vertices = []
dilated_obstacle_list = []
dilated_map = []

#for kalman
mu = np.array([0.,0.,0.,0.,3.14,0.]) #x,y,x_speed,y_speed,angle, angular_speed
motor_cmd = np.array([0,0]) #motor command
thymio_cam_state = np.array([0,0,0]) #ideally pos_x, pos_y, angle

def setup():
    global capture, optimal_path, frame_limits, visibility_graph, vertices, dilated_obstacle_list, dilated_map

    valid_image = False
    while not valid_image:
        valid_image, first_image = capture.read()
    
    thymio_pos,targets_list,obstacles_list,dilated_obstacle_list,dilated_map, frame_limits = object_detection(first_image)

    visibility_graph,start_idx,targets_idx_list,vertices = vis_graph(thymio_pos,targets_list,obstacles_list,dilated_obstacle_list,dilated_map)

    distance_array, path_array = create_distance_path_matrix(visibility_graph,start_idx,[3,1])

    total_distance, targets_idx_order = shortest_path(0, np.array([0]), distance_array)

    #print(total_distance)

    for i in range(len(targets_idx_order)-1):
        for j in range(len(path_array[i][i+1])):
            if i != 0 and j == 0: 
                continue
            #print(path_array[i][i+1][j])
            optimal_path.append(vertices[path_array[i][i+1][j]])

    optimal_path = np.array(optimal_path)

setup()




def cam_thread():
    global capture, optimal_path, frame_limits, visibility_graph, vertices, dilated_obstacle_list, dilated_map, stop_threads

    show_contours = False
    show_polygones = False
    show_dilated_polygones = False
    show_visibility_graph = False
    show_option = [show_contours, show_polygones, show_dilated_polygones, show_visibility_graph]
    while True:

        valid_image, frame = capture.read()
        if stop_threads or not valid_image:
            stop_threads = True
            break

        bounded_frame = frame[frame_limits[0]: frame_limits[1], frame_limits[2]: frame_limits[3]]

        modified_frame = draw_on_frame(bounded_frame, show_option, dilated_obstacle_list, dilated_map)

        cv.imshow('Video', cv.resize(modified_frame, (960, 540)))
        key_pressed = cv.waitKey(1000)
        if key_pressed == ord('q'):
            show_contours = not show_contours
        if key_pressed == ord('w'):
            show_polygones = not show_polygones
        if key_pressed == ord('e'):
            show_dilated_polygones = not show_dilated_polygones
        if key_pressed == ord('r'):
            show_visibility_graph = not show_visibility_graph
        if key_pressed == ord('d'):
            stop_threads = True
            break

        show_option = [show_contours, show_polygones, show_dilated_polygones, show_visibility_graph]

threading.Thread(target=cam_thread).start()






def kalman_thread():
    global mu, thymio_cam_state, motor_cmd
    #Convert speed commands in mm/s
    speed_conv = 0.33478260869565216
    
    #J'essaie avec des valeudrs au bol...
    #mu_init = np.array([374.,1503.,0.,0.,-0.885,0.]) #inital state at zero
    sig_init = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
    T1 = 1/30 #10Hz
    r = 47 #mm
    #uncertainty on state
    R = np.array([[0.01,0.,0.,0.,0.,0.],[0.,0.01,0.,0.,0.,0.],[0.,0.,0.01,0.,0.,0.],[0.,0.,0.,0.01,0.,0.],[0.,0.,0.,0.,0.0000000001,0.],[0.,0.,0.,0.,0.,0.01]])
    #uncertainty on measurement
    Q = np.array([[0.01,0.],[0.,0.01]])

    #move_range = 100

    #hide_camera = 'true'
    camState = 'false' #do not change this one here

    mu_prev = mu
    sig_prev = sig_init

    while True:
        u = speed_conv*motor_cmd
        meas = (thymio_cam_state[0], thymio_cam_state[1])
        (mu,sig) = kalmanFilter(mu_prev, sig_prev, u, meas, T1, r, R, Q, camState)
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
        optimal_path=np.array([[0,0],[50,0],[0,0]])
        await node.lock()
        #pos_r, angle_r, obj_list, prev_err_pos, T, objectif_number
        pos_r = np.array([mu[0][0], mu[1][0]])
        
        angle_r = mu[4]
        #print(pos_r)
        motors_cmd, prev_error, objectif_number = navigation(pos_r, angle_r, optimal_path, prev_error, T1, objectif_number)
        node.send_set_variables(motors_command(motors_cmd))




        async def get_mot_comm(s):   
            await node.wait_for_variables({str(s)})
            return node[s]
        motor_left=await get_mot_comm("motor.left.target")
        motor_right=await get_mot_comm("motor.right.target")
        motor_cmd=np.array([motor_left, motor_right])
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