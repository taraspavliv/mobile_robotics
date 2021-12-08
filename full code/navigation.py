import numpy as np
from numpy.lib.function_base import angle

from tdmclient import ClientAsync
client = ClientAsync()


THRESHOLD_POS=10    #Has to be tuned
THRESHOLD_ANGLE=0.09
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
def reach_obj(pos_r, obj, objectif_number, obj_list):    
            #next_obj(obj_list, objectif_number)
    if on_point(pos_r, obj):
        return True, objectif_number+1
    else:
        return False, objectif_number

def on_point(pos_r, point):
    if pos_r[0]>=point[0]-THRESHOLD_POS and pos_r[0]<=point[0]+THRESHOLD_POS:
        if pos_r[1]>=point[1]-THRESHOLD_POS and pos_r[1]<=point[1]+THRESHOLD_POS:
            return True
    return False

#When on an objectif, choose the optimal side to turn
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
def start_angle(angle_r, dep, obj):
    rad = angle(dep, obj)
    #The angle is good
    if good_angle(angle_r,dep,obj):
        return np.array([SPEED,SPEED])
        #left
    else:
        err_angle=speed_angle_err(angle_r,dep,obj)
        if optimal_side(angle_r, dep, obj):
            motors=np.array([-SPEED,SPEED])
            if abs(err_angle)<0.25:
                motors=np.array([int(-SPEED*err_angle),int(SPEED*err_angle)])
            #err_pos=0
            
            #print(motors)
        else:
            motors=np.array([SPEED,-SPEED])
            if abs(err_angle)<0.25:
                motors=np.array([int(SPEED*err_angle),int(-SPEED*err_angle)])

            #err_pos=0
        
        return motors
        #err_pos


def speed_angle_err(angle_r, dep, obj):
    err_angle=angle_r-angle(dep,obj)
    return err_angle

def good_angle(angle_r,dep,obj):
    rad = angle(dep, obj)
    #The angle is good
#######################################################################################
    if angle_r>0:
        if angle_r>=rad-THRESHOLD_ANGLE and angle_r<=rad+THRESHOLD_ANGLE:
            return True
    else:
        if angle_r>=rad+THRESHOLD_ANGLE and angle_r<=rad-THRESHOLD_ANGLE:
            return True
    return False
#################################################################################

#Modify dep and obj
def next_obj(obj_list, i):
    dep=obj_list[i]
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

    
def navigation(pos_r, angle_r, obj_list, prev_err_pos, T, objectif_number):
    if objectif_number==len(obj_list)-1 :
        motors=np.array([0,0])
        return motors, 0, objectif_number
    else:
        angle_r=angle_r%(np.pi*2)
        
        dep, obj=next_obj(obj_list, objectif_number)
        err_pos=error(pos_r, dep, obj)
        motors=motors_corr(prev_err_pos, err_pos, dep, obj, pos_r, T)
        reached, objectif_number=reach_obj(pos_r, obj, objectif_number, obj_list)


        if reached or (on_point(pos_r, dep)):
            motors=start_angle(angle_r, dep, obj)

    return motors, err_pos, objectif_number