import numpy as np

def SEQTS(cum_r, N, verbose):
    K_arms = len(N)
     
    Theta = [np.inf]*K_arms
    for j in range(K_arms):
        Theta[j] = np.random.beta(cum_r[j]+1,N[j]-cum_r[j]+1)
                    
        # Pick arm if it has maximum upper confidence bound, otherwise pick none
        
    ind = np.argmax(Theta)
        
    if verbose:
        print("cum_r: ",cum_r)
        print("N: ",N)
        print("Theta: ", Theta)
        print("i: ", i, "selected ind: ", ind)
    
    return ind
    
def SEQTS_run(tau,n_arms,R):
    
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
            N[tt,ind[tt]] = 1
            cum_r[tt,ind[tt]] = rewards[tt]
            
            nextInd = SEQTS(np.sum(cum_r,0), np.sum(N,0), False)
    
    regret = [0]*(tau+1)
    for j in range(0,tau):
        ind_t = ind[0:(j+1)*n_arms] #select indices up to time j
        rewards_t = [R[i]-max(R) if i is not None else None for i in ind_t] #get rewards of played arms
        rewards_t = [max(R) if i == 0.0 else i for i in rewards_t] #correct reward if optimal arm is pulled
        rewards_t_clean = [i for i in rewards_t[1:] if i is not None] #remove None pulls
        regret[j+1]=((j+1)*max(R)-sum(rewards_t_clean))
    
    
    return regret, np.sum(N), ind
