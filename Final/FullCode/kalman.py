#Starting with a Kalman using the camera and the motor commands only
import numpy as np
import math

#Next two steps depend on model only

#x is the robot estimation, u are the commands, T1 the time step, r is half the distance between the wheels
def motModel(x, u, T1, r):
    g = np.array([0., 0., 0., 0., 0., 0.]) #just initializing a 2D array
    g[0] = (x[0] + T1*math.cos(x[4])*(u[0] + u[1])/2)
    g[1] = x[1] + T1*math.sin(x[4])*(u[0] + u[1])/2
    g[2] = math.cos(x[4])*(u[0] + u[1])/2
    g[3] = math.sin(x[4])*(u[0] + u[1])/2
    g[4] = x[4] + T1*(u[0] - u[1]) / (2*r)
    g[5] = (u[0] - u[1]) / (2*r)
    return g

#m1 and m2 are motor commands, theta is the current camer estimation, T1 is the time step
def Gjacobian(theta, m1 ,m2, T1):
    G = np.array([[0.,0.,0.,0.,0.,0.], [0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
    G[0,0] = 1
    G[1,1] = 1
    G[0,4] = -T1*math.sin(theta)*(m1+m2)/2
    G[1,4] = T1*math.cos(theta)*(m1+m2)/2
    G[2,4] = -math.sin(theta)*(m1+m2)/2
    G[3,4] = math.cos(theta)*(m1+m2)/2
    G[4,4] = 1
    return G

#Next two steps depend on measurements

def measModel(x, camState):
    h = np.array([0.,0.,0.])
    h[0] = x[0]
    h[1] = x[1]
    h[2] = x[4]
    if camState == False:
        h[0] = 0
        h[1] = 0
        h[2] = 0
    return h

def Hjacobian(camState):
    H = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
    H[0,0] = 1
    H[1,1] = 1
    H[2,4] = 1
    if camState == False:
        H[0,0] = 0
        H[1,1] = 0
        H[2,4] = 0
    return H

def kalmanFilter(mu_prev, sig_prev, u, meas, T1, r, R, Q, camState):
    meas[2] = meas[2]%(2*np.pi)
    #a priori estimations
    mu_est_a_priori = motModel(mu_prev,u,T1,r) #a priori estimation of position. mu_t = g(u,mu_t-1)

    #To avoid errors when the a priori estimation on rotation is close to an angle of 0 / 2*pi
    #we transform the measured angle to be the closest to the estimation
    k = math.floor(mu_est_a_priori[4]/(2*np.pi))
    possible_meas = np.array(list((meas[2] + 2*np.pi*(k+i)) for i in range(-1,2)))
    angles_errors = list(abs(mu_est_a_priori[4] - poss_meas) for poss_meas in possible_meas)
    meas_angle = possible_meas[np.argmin(angles_errors)]
    meas[2] = meas_angle

    theta = mu_est_a_priori[4]
    m1 = u[0]
    m2 = u[1]
    G = Gjacobian(theta,m1,m2,T1)
    sig_est_a_priori = (G @ sig_prev @ np.transpose(G)) + R
    
    #gain computation
    H = Hjacobian(camState)
    K = sig_est_a_priori @ np.transpose(H) @ np.linalg.inv((H @ sig_est_a_priori @np.transpose(H)) + Q)
    
    #measurement update
    mu_est_a_posteriori = mu_est_a_priori.reshape(-1,1) + (K@(meas - measModel(mu_prev, camState)).reshape(-1,1))
    sig_est_a_posteriori = (np.identity(6) - (K @ H)) @ sig_est_a_priori
    
    return mu_est_a_posteriori, sig_est_a_posteriori
