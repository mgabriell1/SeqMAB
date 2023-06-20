import numpy as np

def SR_run(tau,n_arms,R):
    
    from stochastic_env import stochastic_env

    def n_k_f(n,k,K):
        log_bar = 0.5 + sum(1/np.arange(1,K+1)[1:])
        n_k = np.ceil(1/log_bar * (n-K)/(K+1-k))
        return n_k
 
    # Selective reject algorithm
    n_k = [n_k_f(tau,i,n_arms) for i in range(1,n_arms)]
    
    k=0
    reward = np.zeros((tau,n_arms),float)
    avg_reward = [0]*n_arms
    A = np.full((tau,n_arms),0)
    arms_act = np.arange(n_arms)
    ind = [None]*tau
    
    for tt in range(tau):
        A_act = A[:,arms_act]
        reward_act = reward[:,arms_act]
        if all(np.sum(A_act,0) == n_k[k]):
            avg_reward = np.sum(reward_act,0)/np.sum(A_act,0)
            dropped_arm = np.argmin(avg_reward)
            arms_act = np.setdiff1d(arms_act,arms_act[dropped_arm])
            if len(arms_act) > 1:
                k = k + 1
                        
        pos = tt%len(arms_act)
        ii = arms_act[pos]
        reward[tt,ii] = stochastic_env(R,ii)
        A[tt,ii] = 1
        ind[tt] = ii
    
    return arms_act[0], np.sum(A), ind