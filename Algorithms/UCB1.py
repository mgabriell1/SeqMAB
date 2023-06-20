import numpy as np

def UCB1(cum_r, N, t, ran,verbose):
    eps = np.finfo(float).eps
    u = (cum_r)/(N+eps) + np.sqrt(2*np.log10(t)/(N+eps))*ran
    ind = np.argmax(u)

    if verbose:
        print(t)
        print('N ', N)
        print('u ', u)
        print('ind ', ind)
        
    return ind

def UCB1_run(tau,n_arms,R):
    
    from stochastic_env import stochastic_env
    
    N = np.zeros((tau,n_arms),float)
    cum_r = np.zeros((tau,n_arms),float)
    ran = 1
    
    ind = [0] * tau
    rewards = [0] * tau
    
    for tt in range(1,tau):
        
        ind[tt] = UCB1(sum(cum_r), sum(N), tt, ran, False)
    
        rewards[tt] = stochastic_env(R, ind[tt]) 
        N[tt,ind[tt]] = 1
        cum_r[tt,ind[tt]] = rewards[tt]
    
    regret = [0]*(tau+1)
    for j in range(0,tau):
        ind_t = ind[0:j+1] #select indices up to time j
        rewards_t = [R[i]-max(R) for i in ind_t if i is not None] #get rewards of played arms
        rewards_t = [max(R) if i == 0 else i for i in rewards_t] #correct reward if optimal arm is pulled
        regret[j+1]=((j+1)*max(R)-sum(rewards_t[1:]))
        
    # ## Final regret estimate
    # #get rewards of played arms. -2*max(R) is due to the fact that R[i]-1*max(R) is 
    # #the difference between expected rewards and the additional -max(R) is due to the 
    # #fact that the optimal arm is not pulled
    # rewards = [R[i]-2*max(R) for i in ind if i is not None] 
    # #correct reward if optimal arm is pulled
    # rewards = [max(R) if i == -max(R) else i for i in rewards] 
    # regret = (T*max(R)-sum(rewards[1:])) #First index removed as not used
    
    return regret, sum(sum(N)), ind