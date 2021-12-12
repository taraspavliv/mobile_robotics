import numpy as np
from numpy.lib.function_base import angle


THRESHOLD_POS=20    
THRESHOLD_ANGLE=0.05
SLOWDOWN_ANGLE = 0.8
SPEED=100
P=1
D=0.5

#Used to communicate with the Thymio in main
def motors_command(motors):
    return {
        "motor.left.target": [motors[0]],
        "motor.right.target": [motors[1]],
    }


#Returns the angle in 0-360 between two points
def angle(P1,P2):
    angle = np.arctan2(P2[1]-P1[1], P2[0]-P1[0])
    if angle < 0:
        angle = angle + 2*np.pi
    return angle

#Returns True if the Thymio is on the left of its path
def path_side(dep, obj, pos_r):
    v_path = np.array([obj[0]-dep[0], obj[1]-dep[1]])
    v_robot = np.array([pos_r[0]-dep[0], pos_r[1]-dep[1]])
    vect_prod = np.cross(v_path, v_robot)
    #left
    if vect_prod >= 0:
        return True
    #right
    else:
        return False

#Returns the position error
def error(pos_r, dep, obj):
    error = np.linalg.norm(np.cross(obj-dep, dep-pos_r)) / np.linalg.norm(obj-dep)
    return error

#When the objective is reached, one must receive a new objectif and the precedent objective becomes the departure position,
#that's why the objectif number is incremented
def reach_obj(pos_r, obj, objective_number, turn):    
    if np.linalg.norm(pos_r-obj) <= THRESHOLD_POS:
        turn=True
        return True, objective_number+1, turn
    else:
        return False, objective_number, turn

#When on an objectif, chooses the optimal side to turn
#Returns True if the robot angle is bigger than the angle of the dep-obj on a trigonometric circle
def optimal_side(angle_r, dep, obj):
    rad=angle(dep, obj)

    #if robot angle is on the left of the path's one
    if angle_r >= rad:
        if abs(angle_r - rad) <= abs(2*np.pi - angle_r + rad):
            return False
        else:
            return True
    else:
        if abs(angle_r - rad) <= abs(2*np.pi + angle_r - rad):
            return True
        else:
            return False

#When an objectif is reached, the robot turn on itself until it reaches the good angle.
def start_angle(angle_r, dep, obj, turn):
    angle_dep_obj = angle(dep, obj)
    error_angle = angle_err(angle_r, angle_dep_obj)
    if abs(error_angle) <= THRESHOLD_ANGLE:
        #small error, goes on        
        turn=False
        motors=np.array([SPEED,SPEED])
    else:
        if optimal_side(angle_r, dep, obj):
            #turns left
            motors=np.array([-SPEED,SPEED])
            if abs(error_angle)<SLOWDOWN_ANGLE:
                motors=np.array([int(-SPEED*abs(error_angle)/SLOWDOWN_ANGLE),int(SPEED*abs(error_angle)/SLOWDOWN_ANGLE)])
        else:
            #turns right
            motors=np.array([SPEED,-SPEED])
            if abs(error_angle)<SLOWDOWN_ANGLE:
                motors=np.array([int(SPEED*abs(error_angle)/SLOWDOWN_ANGLE),int(-SPEED*abs(error_angle)/SLOWDOWN_ANGLE)])
        turn=True
    return motors, turn

#Returns the angle error between two angles
def angle_err(angle_1, angle_2):
    err_angle = np.arctan2(np.sin(angle_1 - angle_2), np.cos(angle_1 - angle_2))
    return err_angle

#Returns the departure and objectif depending on the objective_number the Thymio has already reached
def next_obj(obj_list, i):
    dep=obj_list[i]
    if i+1 == len(obj_list):
        obj = obj_list[i]
    else:
        obj=obj_list[i+1]
    return dep, obj

#PD regulator to correct the position when the Thymio is rolling between two objectives
def motors_corr(prev_err_pos, err_pos, dep, obj, pos_r, T):

    speed_corr=err_pos*P + D*(err_pos-prev_err_pos)/T
    motors=np.array([0,0])

    if path_side(dep, obj, pos_r):
        #On the left side of the segment, should turn right
        motors[0]=SPEED+speed_corr
        motors[1]=SPEED-speed_corr
    else:
        #On the right side of the segment, should turn left
        motors[0]=SPEED-speed_corr
        motors[1]=SPEED+speed_corr

    return motors

#Returns the motors commands one should send to the Thymio 
def navigation(pos_r, angle_r, obj_list, prev_err_pos, T, objective_number, turn):
    #Stops the Thymio at the end of its path
    if objective_number==len(obj_list)-1 :
        motors=np.array([0,0])
        turn = False
        return motors, 0, objective_number, turn
        
    else:
        #Translates the angle in [0,2pi]
        angle_r=angle_r%(np.pi*2)
        
        dep, obj=next_obj(obj_list, objective_number)
        err_pos=error(pos_r, dep, obj)

        if not turn:
            motors=motors_corr(prev_err_pos, err_pos, dep, obj, pos_r, T)
            reached, objective_number, turn=reach_obj(pos_r, obj, objective_number, turn)
            #If an objective is reached, updates the objective and turns until the angle is good
            if reached :
                dep, obj=next_obj(obj_list, objective_number)
                motors, turn = start_angle(angle_r, dep, obj, turn)
        else:
            motors,turn = start_angle(angle_r, dep, obj, turn)

    return motors, err_pos, objective_number, turn