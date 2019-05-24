import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
import bigfloat
from decimal import *
from pytictoc import TicToc


def build_shadow_M (X,tau,E,T):
    '''Build the shadow manifold of the time series signal X, with E variables and sampling tau'''
    shadow_M=np.zeros((T-E+1,E))
    for i in range((tau*E-1),T):
        sample=np.zeros((E))
        for j in range(0,E):
            sample[j]=X[i-j*tau]
        shadow_M[i-(tau*E-1),:]=sample
    return shadow_M

def build_shadow_selfpred (X,tau,E,T,tpred):
    '''Build the shadow manifold of the time series signal X, with E variables and sampling tau and the shadow manifold of the time series signal X+tpred to perform self prediction'''
    shadow_X=np.zeros((T-E+1-tpred,E))
    for i in range((tau*E-1),T-tpred):
        sample=np.zeros((E))
        for j in range(0,E):
            sample[j]=X[i-j*tau]
        shadow_X[i-(tau*E-1),:]=sample
    
    shadow_Xpred=np.zeros((T-E+1-tpred,E))
    for i in range((tau*E+tpred-1),T):
        sample=np.zeros((E))
        for j in range(0,E):
            sample[j]=X[i-j*tau]
        shadow_Xpred[i-(tau*E-1)-tpred,:]=sample
    
    return shadow_X,shadow_Xpred

def sample_manifold (M, L):
    '''Randomly select L points from the shadow manifold M'''
    new_M=np.zeros((L,M.shape[1]))
    idx=np.random.randint(M.shape[0], size=L)
    for i in range(L):
        new_M[i,:]=M[idx[i],:]
    return new_M, idx

def nearest_points(shadow_x,M,idx,E):
    '''Find the E+2 nearest points to each point in the reconstructed manifold, it is only necessary E+1 
    but the first one is the point it self. The distance provided is the euclidean distance used to compute
    the weights in the weighted average for estimation.'''
    
    distances_t=np.zeros((len(shadow_x),E+1))
    indices_t=np.zeros((len(shadow_x),E+1),dtype=int)
    
    for i in range(len(shadow_x)):
        if shadow_x[i] in M:
            nbrs=NearestNeighbors(n_neighbors=E+2,algorithm='kd_tree',metric='euclidean').fit(M)
            distances, indices=nbrs.kneighbors(M)
            k,l = np.where(M == shadow_x[i,:])
            distances_t[i,:]=distances[k[0],1:E+3]
            indices_t[i,:]=indices[k[0],1:E+3]
        else:
            new_M=np.concatenate((M,[shadow_x[i,:]]))
            nbrs=NearestNeighbors(n_neighbors=E+2,algorithm='kd_tree',metric='euclidean').fit(new_M)
            distances, indices=nbrs.kneighbors(new_M)
            distances_t[i,:]=distances[-1,1:E+3]
            indices_t[i,:]=indices[-1,1:E+3]
    
    
    for i in range(len(indices_t)):
        for j in range(len(indices_t[i])):
            indices_t[i,j]=idx[indices_t[i,j]]
    return distances_t, indices_t

def compute_weights(shadow,distances,indices,T,E,eps=1e-4):
    weights=np.zeros((len(shadow),E+1))
    weights_u=np.zeros((len(shadow),E+1))
    MY_pred=np.zeros((len(shadow),E))
    for i in range (len(shadow)):
        for j in range(0,E+1):
            num=(distances[i,j])
            den=(eps+distances[i,1])
            weights_u[i,j-1]=np.exp(-(num)/(den))
        if np.isinf(np.sum(weights[i,:])):
            weights[i,0]=1
        else:
            weights[i,:]=weights_u[i,:]/np.sum(weights_u[i,:])
        MY_pred[i,:]=np.dot(weights[i,:],(list(shadow[j,:] for j in indices[i,:])))
    return weights, MY_pred

def compute_prediction(shadow_y,weights,E,tau,T,indices):
    MY_pred=np.zeros((len(shadow_y),E))
    for i in range(len(shadow_y)):
        for j in range(0,E+1):
            MY_pred[i,:]=MY_pred[i,:]+weights[i,j]*shadow_y[indices[i,j],:]
    y_pred=MY_pred[:,0]
    y_target=shadow_y[:,0]
    return MY_pred, shadow_y, y_pred, y_target

def compute_corr(y_pred, y_target):
    corr=np.corrcoef(y_pred,y_target)[1,0]
    return corr

def compute_xmap(X,Y,T,E,tau,L):
    '''Compute the convergent cross mapping between X and Y'''
    time=TicToc()
    
    # Build the shadow manifold 
    print('Build shadow manifold')
    time.tic()
    shadow_x=build_shadow_M(X,tau,E,T)
    shadow_y=build_shadow_M(Y,tau,E,T)
    time.toc()
    
    # Select randomly L points from the shadow manifold 
    print('Select L points from shadow manifold')
    time.tic()
    recon_Mx, idx_x=sample_manifold(shadow_x,L)
    recon_My, idx_y=sample_manifold(shadow_y,L)  
    time.toc()
    
    ########## Predict Y from X ##########################
    print('Starting prediction of Y from X...')
    
    # find nearest neighbors
    print('Finding nearest neighbors...')
    time.tic()
    distances_x, indices_x=nearest_points(shadow_x,recon_Mx,idx_x,E)
    time.toc()
    
    # compute weights
    print('Compute weights and prediction')
    time.tic()
    weights_w_x, My_pred=compute_weights(shadow_y,distances_x,indices_x,T,E)
    y_pred=My_pred[:,0]
    y_target=shadow_y[:,0]
    time.toc()
    # compute prediction
    #My_pred,My_target,y_pred, y_target=compute_prediction(shadow_y,weights_w_x,E,tau,T,indices_x)
    
    ########## Predict X from Y ##########################
    print('Starting prediction of X from Y...')
    # find nearest neighbors
    print('Finding nearest neighbors...')
    time.tic()
    distances_y, indices_y=nearest_points(shadow_y, recon_My,idx_y,E)
    time.toc()
    
    # compute weights
    print('Compute weights and prediction')
    time.tic()
    weights_w_y, Mx_pred=compute_weights(shadow_x,distances_y,indices_y,T,E)
    x_pred=Mx_pred[:,0]
    x_target=shadow_x[:,0]
    time.toc()
    
    # compute prediction 
    #Mx_pred,Mx_target,x_pred, x_target=compute_prediction(shadow_x,weights_w_y,E,tau,T,indices_y)
    
    return y_pred, y_target, x_pred, x_target

def compute_xmap_selfpred(X,T,E,tau,L,tpred):
    '''Compute the convergent cross mapping between X and Y'''
    
    # Build the shadow manifold 
    shadow_x,shadow_x_pred=build_shadow_selfpred(X,tau,E,T,tpred)
    
    # Select randomly L points from the shadow manifold 
    recon_Mx, idx_x=sample_manifold(shadow_x,L)
    recon_My, idx_y=sample_manifold(shadow_x_pred,L)  
    
    ########## Predict Y from X ##########################
    
    # find nearest neighbors
    distances_x, indices_x=nearest_points(shadow_x,recon_Mx,idx_x,E)
    
    # compute weights
    weights_w_x=compute_weights(distances_x,indices_x,T,E)
    
    # compute prediction
    My_pred,My_target,y_pred, y_target=compute_prediction(shadow_x_pred,weights_w_x,E,tau,T,indices_x)
    
    ########## Predict X from Y ##########################
    
    # find nearest neighbors
    distances_y, indices_y=nearest_points(shadow_x_pred, recon_My,idx_y,E)
    
    # compute weights
    weights_w_y=compute_weights(distances_y,indices_y,T,E)
    
    # compute prediction 
    Mx_pred,Mx_target,x_pred, x_target=compute_prediction(shadow_x,weights_w_y,E,tau,T,indices_y)
    
    return y_pred, y_target, x_pred, x_target













