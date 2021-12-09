import numpy as np
from numpy.lib.function_base import angle

from tdmclient import ClientAsync
client = ClientAsync()


THRESHOLD_POS=20    #Has to be tuned
THRESHOLD_ANGLE=0.05
SLOWDOWN_ANGLE = 0.8
SPEED=100
P=1
D=0.5


#Navigation

#Output Kalman: Position, angle, vitesse, vitesse angulaire

#Output path planning: Prochaine position

def motors_command(motors):
    return {
        "motor.left.target": [motors[0]],
        "motor.right.target": [motors[1]],
    }



def angle(P1,P2):
    angle=np.arctan2(P2[1]-P1[1],P2[0]-P1[0])
    if angle<0:
        angle=angle+2*np.pi
    return angle


def path_side(dep, obj, pos_r):
    v_path = np.array([obj[0]-dep[0],obj[1]-dep[1]])
    v_robot = np.array([pos_r[0]-dep[0],pos_r[1]-dep[1]])
    vect_prod=-np.cross(v_path,v_robot)
    #left
    if vect_prod>=0:
        return True
    #right
    else:
        return False
#return the position error
def error(pos_r, dep, obj):
    error=np.linalg.norm(np.cross(obj-dep, dep-pos_r))/np.linalg.norm(obj-dep)
    return error

#When the objectif is reached, one must receive a new objectif and the precedent objectif becomes the depart position
def reach_obj(pos_r, obj, objectif_number,turn):    
    if np.linalg.norm(pos_r-obj) <= THRESHOLD_POS:
        turn=True
        return True, objectif_number+1,turn
    else:
        return False, objectif_number,turn

#When on an objectif, choose the optimal side to turn
#returns true if left is closer
def optimal_side(angle_r, dep, obj):
    rad=angle(dep,obj)

    #if robot angle is on the left
    if angle_r>=rad:
        #right
        if abs(angle_r-rad) <= abs(2*np.pi-angle_r+rad):
            return False
        else:
            return True
    else:
        #right
        if abs(angle_r-rad) <= abs(2*np.pi+angle_r-rad):
            return True
        else:
            return False

#When an objectif is reached, the robot turn on itself until it reaches the good angle.
def start_angle(angle_r, dep, obj, turn):
    #The angle is good
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
            #err_pos=0
            
        else:
            #turns right
            motors=np.array([SPEED,-SPEED])
            if abs(error_angle)<SLOWDOWN_ANGLE:
                motors=np.array([int(SPEED*abs(error_angle)/SLOWDOWN_ANGLE),int(-SPEED*abs(error_angle)/SLOWDOWN_ANGLE)])

            #err_pos=0
        turn=True
    return motors, turn


def angle_err(angle_1, angle_2):
    err_angle = np.arctan2(np.sin(angle_1 - angle_2), np.cos(angle_1 - angle_2)) #black magic formula to give angle error
    return err_angle

#Modify dep and obj
def next_obj(obj_list, i):
    dep=obj_list[i]
    if i+1 == len(obj_list):
        obj = obj_list[i]
    else:
        obj=obj_list[i+1]
    return dep, obj



#Je pense qu'un regulateur P devrait suffire....
def motors_corr(prev_err_pos, err_pos, dep, obj, pos_r, T):

    speed_corr=err_pos*P + D*(err_pos-prev_err_pos)/T
    motors=np.array([0,0])

    if path_side(dep, obj, pos_r):
        #left
        #print("Tourne à gauche")
        motors[0]=SPEED-speed_corr
        motors[1]=SPEED+speed_corr
    else:
        #print("Tourne à droite")
        motors[0]=SPEED+speed_corr
        motors[1]=SPEED-speed_corr
    

    return motors

    
def navigation(pos_r, angle_r, obj_list, prev_err_pos, T, objectif_number, turn):
    if objectif_number==len(obj_list)-1 :
        motors=np.array([0,0])
        turn = False
        return motors, 0, objectif_number, turn
    else:
        angle_r=angle_r%(np.pi*2)
        
        dep, obj=next_obj(obj_list, objectif_number)
        err_pos=error(pos_r, dep, obj)

        angle_dep_obj = angle(dep, obj)
        error_angle = angle_err(angle_r, angle_dep_obj)

        if not turn:    
            motors=motors_corr(prev_err_pos, err_pos, dep, obj, pos_r, T)
            reached, objectif_number, turn=reach_obj(pos_r, obj, objectif_number, turn)
            if reached :
                dep, obj=next_obj(obj_list, objectif_number)
                motors, turn = start_angle(angle_r, dep, obj, turn)
        else:
            motors,turn = start_angle(angle_r, dep, obj, turn)


    return motors, err_pos, objectif_number, turn