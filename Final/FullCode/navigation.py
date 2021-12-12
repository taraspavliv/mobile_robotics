"""
navigation.py

This file is used to ensure that the Thymio will go to a list of 2D coordinates in the right order
"""

import numpy as np
from numpy.lib.function_base import angle

THRESHOLD_POS=20    
THRESHOLD_ANGLE=0.05
SLOWDOWN_ANGLE = 0.8
SPEED=100
P=1
D=0.5

def motors_command(motors):
    """
    motors_command creates the structure used to communicate with the Thymio

    :motors:    The index 0 is the value calculated for the left motor 
                and the index 1, for the right motor

    :return:    The structure to communicate with the Thymio
    """

    return {
        "motor.left.target": [motors[0]],
        "motor.right.target": [motors[1]],
    }


def angle(P1,P2):
    """
    angle calculates the angle between the line given by two 2D points and the horizontal axis. 
    The angles are given between 0 and 2pi, 0 being the horizontal axis

    :P1:        The coordinates of the first point being the origin of the axis
    :P2:        The second point to calculate the angle

    :return:    The angle in [0,2pi]
    """
    angle = np.arctan2(P2[1]-P1[1], P2[0]-P1[0])
    if angle < 0:
        angle = angle + 2*np.pi
    return angle


def path_side(dep, obj, pos_r):
    """
    path_side returns True if the Thymio is on the left of its path and False otherwise

    :dep:   The departure point of the Thymio
    :obj:   The objective point of the Thymio
    :pos_r: The position of the robot

    :return: A boolean. False if the Thymio is on the right of its path, True otherwise
    """
    v_path = np.array([obj[0]-dep[0], obj[1]-dep[1]])
    v_robot = np.array([pos_r[0]-dep[0], pos_r[1]-dep[1]])
    vect_prod = np.cross(v_path, v_robot)
    #left
    if vect_prod >= 0:
        return True
    #right
    else:
        return False


def error(pos_r, dep, obj):
    """
    error calculates the perpendicular distance between the line formed by the two 2D points dep and obj 
    and the position of the robot

    :pos_r:     The position of the robot
    :dep:       The departure point of the Thymio
    :obj:       The objective point of the Thymio

    :return:    error, the distance from the position of the robot to the line
    """
    error = np.linalg.norm(np.cross(obj-dep, dep-pos_r)) / np.linalg.norm(obj-dep)
    return error


def reach_obj(pos_r, obj, objective_number, turn):
    """
    reach_obj detects when the Thymio is in the circle of center obj and radius THRESHOLD_POS. This is used to update
    the number of objective reached and the Thymio state

    :pos_r:             The position of the robot
    :obj:               The objective point of the Thymio
    :objective_number:  The number of objective the Thymio has already reached
    :turn:              The state of the Thymio to know if the Thymio is moving forward (False) or turning (True)   

    :return:            reached, boolean as True if the Thymio arrives on its objective, False otherwise
                        objective_number, the number of objective already reached
                        turn, the state of the Thymio to know if the Thymio is moving forward (False) or turning (True)
    """
    if np.linalg.norm(pos_r-obj) <= THRESHOLD_POS:
        turn = True
        reached = True
        return reached, objective_number+1, turn
    else:
        reached = False
        return reached, objective_number, turn


def optimal_side(angle_r, dep, obj):
    """
    optimal_side calculates on which side it is better to turn when an objective is reached

    :angle_r:   The angle of the robot in [0,2pi]
    :dep:       The departure point of the Thymio
    :obj:       The objective point of the Thymio

    :return:    A boolean, True if the Thymio has to turn to the left
    """

    rad=angle(dep, obj)

    #if robot angle is bigger than the path's one
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


def start_angle(angle_r, dep, obj, turn):
    """
    start_angle ensures that the Thymio continues to turn until the angle is good enough when an objective is reached.
    The rotation speed starts decreasing when the angle SLOWDOWN_ANGLE is reached.

    :angle_r:   The angle of the robot in [0,2pi]
    :dep:       The departure point of the Thymio
    :obj:       The objective point of the Thymio
    :turn:      The state of the Thymio to know if the Thymio is moving forward (False) or turning (True)

    :return:    motors, the speed commands for the motors
                turn, the state of the Thymio to know if the Thymio is moving forward (False) or turning (True)
    """
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


def angle_err(angle_1, angle_2):
    """
    angle_err calculates the error between two angles

    :angle1:    The first angle
    :angle2:    The second angle

    :return:    err_angle, the error between two angles
    """
    err_angle = np.arctan2(np.sin(angle_1 - angle_2), np.cos(angle_1 - angle_2))
    return err_angle


def next_obj(obj_list, i):
    """
    next_obj returns the departure and objective depending on the objective_number the Thymio has already reached

    :obj_list:  The list of the points to reach
    :i:         The number of objective the Thymio has already reached

    :return:    dep, the departure point
                obj, the objective point
    """
    dep=obj_list[i]
    if i+1 == len(obj_list):
        obj = obj_list[i]
    else:
        obj=obj_list[i+1]
    return dep, obj


def motors_corr(prev_err_pos, err_pos, dep, obj, pos_r, T):
    """
    motors_corr calculates the speed correction using a PD regulator to correct the position when the
    Thymio is rolling between two objectives

    :prev_err_pos:  The previous position error
    :err_pos:       The position error
    :dep:           The departure point of the Thymio
    :obj:           The objective point of the Thymio
    :pos_r:         The position of the robot
    :T:             The time between to calls of this function

    :return:        motors, the speed commands for the motors
    """

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


def navigation(pos_r, angle_r, obj_list, prev_err_pos, T, objective_number, turn):
    """
    navigation returns the motors commands one should send to the Thymio to ensure the good displacement 
    of the Thymio on the map

    :pos_r:             The position of the robot
    :angle_r:           The angle of the robot in [-Inf, Inf]
    :obj_list:          The list of the points to reach   
    :prev_err_pos:      The previous position error
    :T:                 The time between to calls of this function
    :objective_number:  The number of objective the Thymio has already reached
    :turn:              The state of the Thymio to know if the Thymio is moving forward (False) or turning (True)

    :return:            motors, the speed commands for the motors
                        err_pos, the position error
                        objective_number, the number of objective already reached
                        turn, the state of the Thymio to know if the Thymio is moving forward (False) or turning (True)
    """

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