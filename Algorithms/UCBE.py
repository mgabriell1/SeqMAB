import numpy as np

def UCBE(cum_r, N, a):
    
    n_arms = len(N)
    B = [0]*n_arms
    
    if any(N==0):
        ind = np.argmin(N)
        
    else:
        for i in range(n_arms):
            B[i] = cum_r[i]/N[i]+np.sqrt(a/N[i])
        
        ind = np.argmax(B)
    
    return ind

def UCBE_run(tau,n_arms,R,c):
    
    from stochastic_env import stochastic_env
    
    N = np.zeros((tau,n_arms),float)
    cum_r = np.zeros((tau,n_arms),float)
    
    ind = [0] * tau
    rewards = [0] * tau
     
    Delta2 = (R-max(R))**2
    Delta2 = np.delete(Delta2,np.argmin(Delta2))
    H1 = sum(1/Delta2)
    #a = 25/36 * (tau-n_arms)/H1
    a = c*tau/H1 
    
    for tt in range(1,tau):
        
        ind[tt] = UCBE(np.sum(cum_r,0), np.sum(N,0), a)
    
        rewards[tt] = stochastic_env(R, ind[tt]) 
        N[tt,ind[tt]] = 1
        cum_r[tt,ind[tt]] = rewards[tt]
        
    armsEstimates = np.sum(cum_r,0)/np.sum(N,0)
    bestArm = np.argmax(armsEstimates)
    #correct = bestArm == np.argmax(R)
    #error = max(R) - R[bestArm]
    
    return bestArm, np.sum(N), ind