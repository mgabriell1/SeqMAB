import numpy as np

def SEQUCBE(cum_r, N, a):
    
    n_arms = len(N)
    B = [0]*n_arms
        
    if any(N==0):
        ind = np.argmin(N)
        
    else:
        for i in range(n_arms):
            B[i] = cum_r[i]/N[i]+np.sqrt(a/N[i])
        
        ind = np.argmax(B)
    
    return ind

def SEQUCBE_run(tau,n_arms,R,c,pullsMax):
    
    from stochastic_env import stochastic_env
    
    T = tau*n_arms
    
    N = np.zeros((T,n_arms),float)
    cum_r = np.zeros((T,n_arms),float)
    
    ind = [None] * T
    rewards = [0] * T
     
    Delta2 = (R-max(R))**2
    Delta2 = np.delete(Delta2,np.argmin(Delta2))
    H1 = sum(1/Delta2)
    #a = 25/36 * (T-n_arms)/H1
    a = c*tau/H1 
    
    nextInd = 0
    
    for tt in range(1,T):
        i = np.mod(tt-1,n_arms)
        
        if nextInd == i:  
            ind[tt] = i
            rewards[tt] = stochastic_env(R, ind[tt]) 
            N[tt,ind[tt]] = 1
            cum_r[tt,ind[tt]] = rewards[tt]
            
            nextInd = SEQUCBE(np.sum(cum_r,0), np.sum(N,0), a)
        
        if np.sum(N) >= pullsMax:
            break
        
    armsEstimates = np.sum(cum_r,0)/np.sum(N,0)
    bestArm = np.argmax(armsEstimates)
    
    return bestArm, np.sum(N), ind, tt/n_arms

def SEQUCBE_NoLim_run(tau,n_arms,R,c):
    
    from stochastic_env import stochastic_env
    
    T = tau*n_arms
    
    N = np.zeros((T,n_arms),float)
    cum_r = np.zeros((T,n_arms),float)
    
    ind = [None] * T
    rewards = [0] * T
     
    Delta2 = (R-max(R))**2
    Delta2 = np.delete(Delta2,np.argmin(Delta2))
    H1 = sum(1/Delta2)
    #a = 25/36 * (T-n_arms)/H1
    a = c*tau/H1 
    
    nextInd = 0
    
    for tt in range(1,T):
        i = np.mod(tt-1,n_arms)
        
        if nextInd == i:  
            ind[tt] = i
            rewards[tt] = stochastic_env(R, ind[tt]) 
            N[tt,ind[tt]] = 1
            cum_r[tt,ind[tt]] = rewards[tt]
            
            nextInd = SEQUCBE(np.sum(cum_r,0), np.sum(N,0), a)
        
        
    armsEstimates = np.sum(cum_r,0)/np.sum(N,0)
    bestArm = np.argmax(armsEstimates)

    
    return bestArm, np.sum(N), ind
