import numpy as np

def SEQUCB1(cum_r, N, t, ran,verbose):
    eps = np.finfo(float).eps
    K_arms = len(N)
    U = [np.inf]*K_arms
    for j in range(K_arms):
        U[j] = cum_r[j]/(N[j]+eps) + np.sqrt(2*np.log(np.sum(N)+eps)/(N[j]+eps))*ran
        
    ind = np.argmax(U)

    if verbose:
        print(t)
        print('N ', N)
        print('u ', U)
        print('ind ', ind)
        
    return ind

def SEQUCB1_run(tau,n_arms,R):
    
    from stochastic_env import stochastic_env
    
    T = tau*n_arms
    N = np.zeros((T,n_arms),float)
    cum_r = np.zeros((T,n_arms),float)
    ran = 1
    
    ind = [None] * T
    rewards = [0] * T 
    
    nextInd = 0
    
    for tt in range(1,T):
        i = np.mod(tt-1,n_arms)

        if nextInd == i:
            ind[tt] = i
            rewards[tt] = stochastic_env(R, ind[tt]) 
            N[tt,nextInd] = 1
            cum_r[tt,nextInd] = rewards[tt]
        
            nextInd = SEQUCB1(np.sum(cum_r,0), np.sum(N,0), tt, ran, False)

    
    regret = [0]*(tau+1)
    for j in range(0,tau):
        ind_t = ind[0:(j+1)*n_arms] #select indices up to time j
        rewards_t = [R[i]-max(R) if i is not None else None for i in ind_t] #get rewards of played arms
        rewards_t = [max(R) if i == 0.0 else i for i in rewards_t] #correct reward if optimal arm is pulled
        rewards_t_clean = [i for i in rewards_t[1:] if i is not None] #remove None pulls
        regret[j+1]=((j+1)*max(R)-sum(rewards_t_clean))
    
    
    return regret, np.sum(N), ind




