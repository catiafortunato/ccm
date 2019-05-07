import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
import bigfloat
from decimal import *

def build_shadow_M (X,tau,E,T):
    '''Build the shadow manifold of the time series signal X, with E variables and sampling tau (how far back are we looking). 
    Returns the shadow manifold'''
    shadow_M=np.zeros((T-E+1,E))
    for i in range((tau*E-1),T):
        sample=np.zeros((E))
        for j in range(0,E):
            sample[j]=X[i-j*tau]
        shadow_M[i-(tau*E-1),:]=sample
    return shadow_M

def sample_manifold (M, L):
    '''Randomly select L points from the shadow manifold M'''
    new_M=np.zeros((L,M.shape[1]))
    idx=np.random.randint(M.shape[0], size=L)
    for i in range(L):
        new_M[i,:]=M[idx[i],:]
    return new_M, idx

def nearest_points(M,idx,E):
    '''Find the E+2 nearest points to each point in the reconstructed manifold, it is only necessary E+1 
    but the first one is the point it self. The distance provided is the euclidean distance used to compute
    the weights in the weighted average for estimation.'''
    
    nbrs=NearestNeighbors(n_neighbors=E+2,algorithm='kd_tree',metric='euclidean').fit(M)
    distances, indices=nbrs.kneighbors(M)
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            indices[i,j]=idx[indices[i,j]]
    return distances, indices

def compute_weights(distances,indices,L,E,eps=1e-4):
    weights=np.zeros((L,E+1))
    weights_u=np.zeros((L,E+1))
    for i in range (L):
        for j in range(1,E+2):
            num=(distances[i,j])
            den=(eps+distances[i,1])
            weights_u[i,j-1]=bigfloat.exp(-(num)/(den))
        if np.isinf(np.sum(weights[i,:])):
            weights[i,0]=1
        else:
            weights[i,:]=weights_u[i,:]/np.sum(weights_u[i,:])
    return weights

def compute_prediction(Mx,shadow_y,weights,E,tau,L,indices):
    MY_pred=np.zeros((L,E))
    for i in range(L):
        for j in range(1,E+2):
            MY_pred[i,:]=MY_pred[i,:]+weights[i,j-1]*shadow_y[indices[i,j],:]
    MY_target=np.zeros((L,E))
    for l in range(L):
        MY_target[l,:]=shadow_y[indices[l,0],:]
    y_pred=MY_pred[:,0]
    y_target=MY_target[:,0]
    return MY_pred, MY_target, y_pred, y_target

def compute_corr(y_pred, y_target):
    corr=np.corrcoef(y_pred,y_target)[1,0]
    return corr

def compute_xmap(X,Y,T,E,tau,L):
    
    # Build the shadow manifold 
    shadow_x=build_shadow_M(X,tau,E,T)
    shadow_y=build_shadow_M(Y,tau,E,T)
    
    # Select randomly L points from the shadow manifold 
    recon_Mx, idx_x=sample_manifold(shadow_x,L)
    recon_My, idx_y=sample_manifold(shadow_y,L)  
    
    ########## Predict Y from X ##########################
    
    # find nearest neighbors
    distances_x, indices_x=nearest_points(recon_Mx,idx_x,E)
    
    # compute weights
    weights_w_x=compute_weights(distances_x,indices_x,L,E)
    
    # compute prediction
    My_pred,My_target,y_pred, y_target=compute_prediction(recon_Mx,shadow_y,weights_w_x,E,tau,L,indices_x)
    
    ########## Predict X from Y ##########################
    
    # find nearest neighbors
    distances_y, indices_y=nearest_points(recon_My,idx_y,E)
    
    # compute weights
    weights_w_y=compute_weights(distances_y,indices_y,L,E)
    
    # compute prediction 
    Mx_pred,Mx_target,x_pred, x_target=compute_prediction(recon_My,shadow_x,weights_w_y,E,tau,L,indices_y)
    
    return y_pred, y_target, x_pred, x_target















