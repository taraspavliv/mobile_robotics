#Starting with a Kalman using the camera and the motor commands only
import numpy as np
import math
import random
import matplotlib.pyplot as plt

#Next two steps depend on model only

def motModel(x, u, T1, r):
    g = np.array([0., 0., 0., 0., 0., 0.]) #just initializing a 2D array
    g[0] = (x[0] + T1*math.cos(x[4])*(u[0] + u[1])/2)
    g[1] = x[1] + T1*math.sin(x[4])*(u[0] + u[1])/2
    g[2] = math.cos(x[4])*(u[0] + u[1])/2
    g[3] = math.sin(x[4])*(u[0] + u[1])/2
    g[4] = x[4] + T1*(u[0] - u[1]) / (2*r)
    g[5] = (u[0] - u[1]) / (2*r)
    return g

def Gjacobian(theta, m1 ,m2, T1, r):
    #G = np.array([[0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.]])
    #the previous one was 6x8
    #i need 6x6
    G = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
    G[0,0] = 1
    G[1,1] = 1
    G[0,4] = -T1*math.sin(theta)*(m1+m2)/2
    G[1,4] = T1*math.cos(theta)*(m1+m2)/2
    G[2,4] = -math.sin(theta)*(m1+m2)/2
    G[3,4] = math.cos(theta)*(m1+m2)/2
    G[4,4] = 1
    return G

#Next two steps depend on measurements

def measModel(x,T1,r):
    h = np.array([0.,0.])
    h[0] = x[0]
    h[1] = x[1]
    return h

def Hjacobian(theta, T1, camState):
    H = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
    H[0,0] = 1
    H[1,1] = 1
    if camState == False:
        H[0,0] = 0
        H[1,1] = 0
    return H

def kalmanFilter(mu_prev, sig_prev, u, meas, T1, r, R, Q, camState):
    
    #a priori estimations
    mu_est_a_priori = motModel(mu_prev,u,T1,r) #a priori estimation of position. mu_t = g(u,mu_t-1)
    
    theta = mu_est_a_priori[4]
    #print('theta a priori', theta)
    m1 = u[0]
    m2 = u[1]
    G = Gjacobian(theta,m1,m2,T1,r)
    sig_est_a_priori = (G @ sig_prev @ np.transpose(G)) + R
    
    #gain computation
    H = Hjacobian(theta,T1, camState)
    K = sig_est_a_priori @ np.transpose(H) @ np.linalg.inv((H @ sig_est_a_priori @np.transpose(H)) + Q)
    
    #measurement update
    mu_est_a_posteriori = mu_est_a_priori.reshape(-1,1) + (K@(meas - measModel(mu_prev,T1,r)).reshape(-1,1))
    sig_est_a_posteriori = (np.identity(6) - (K @ H)) @ sig_est_a_priori
    #print('theta a posteriori', mu_est_a_posteriori[4])
    #print('')
    
    return mu_est_a_posteriori, sig_est_a_posteriori
    #return mu_est_a_priori, sig_est_a_priori
