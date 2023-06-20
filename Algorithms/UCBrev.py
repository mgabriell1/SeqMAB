import numpy as np
    
def UCBrev_run(T,n_arms,R):
    
    from stochastic_env import stochastic_env
    def n_m(deltam,rounds):
        n_m = np.ceil(2*np.log(rounds*deltam**2)/(deltam**2))
        return n_m
    
    k=0
    deltam=1
    A = np.full((T,n_arms),1,dtype=int)
    N_tot = [0]*n_arms
    N = [0]*n_arms
    reward = np.zeros((T,n_arms),float)
    avg_reward = [0]*n_arms
    tabu = []
    for tt in range(1,T):
        A[tt,] = A[tt-1,]
        if all(i==n_m(deltam,T) for i in N):        
            if sum(A[tt,])>1:
                for ii in range(n_arms):
                    if A[tt,ii]==1:
                        avg_reward[ii]=np.mean(reward[1:tt,ii])
                    else:
                        tabu.append(avg_reward[ii])
                
                avg_reward_possible = np.setdiff1d(avg_reward,tabu)
                bound = np.sqrt(np.log(T*deltam**2)/(2*n_m(deltam,T)))
                min_pos = np.where(avg_reward+bound<max(avg_reward)-bound)
                           
                A[tt,min_pos] = 0
                deltam = deltam/2
        
        for ii in range(n_arms):
            if A[tt,ii] == 0:
                reward[tt,ii] = 0
            else:
                reward[tt,ii] = stochastic_env(R,ii) #gaussian_env(R,R_sd,ii)
                N_tot[ii] = N_tot[ii] + 1
                    
        N = [N_tot[i] for i in range(n_arms) if A[tt,i]==1]
            
    
    opt_arm = np.argmax(R)
    regret = [0]*(T+1)
    for j in range(T):
        rewards_t = (A[0:j,]*R).flatten()
        rewards_t = [k for k in rewards_t if k > 0]
        rewards_t = [k - max(R) for k in rewards_t]
        rewards_t = [k if k < 0 else max(R) for k in rewards_t]
        regret[j+1]=(j*max(R)-sum(rewards_t))
        
    Aout = np.where(A==0,np.nan,A)
    Aout = (Aout*np.arange(n_arms)).ravel()
    Aout = np.where(Aout==np.nan,None,Aout)
    return regret,np.sum(A), np.sum(A,1), Aout

# def UCBrev_run(T,n_arms,R):
    
#     from stochastic_env import stochastic_env
#     def n_m(deltam,rounds):
#         n_m = np.ceil(2*np.log(rounds*deltam**2)/(deltam**2))
#         return n_m
    
#     deltam=1
#     A = np.full((T,n_arms),1,dtype=int)
#     N_tot = [0]*n_arms
#     N = [0]*n_arms
#     reward = np.zeros((T,n_arms),float)
#     avg_reward = [0]*n_arms
#     tabu = []
#     for tt in range(1,T):
#         A[tt,] = A[tt-1,]
#         if all(i==n_m(deltam,T) for i in N):                    
#             if sum(A[tt,])>1:
#                 for ii in range(n_arms):
#                     if A[tt,ii]==1:
#                         avg_reward[ii]=np.mean(reward[1:tt,ii])
#                     else:
#                         tabu.append(avg_reward[ii])
                
#                 bound = np.sqrt(np.log(T*deltam**2)/(2*n_m(deltam,T)))
#                 min_pos = np.where(avg_reward+bound<max(avg_reward)-bound)
                           
#                 A[tt,min_pos] = 0
#                 deltam = deltam/2
#                 #print(tt, A[tt,])
        
#         for ii in range(n_arms):
#             if A[tt,ii] == 0:
#                 reward[tt,ii] = None
#             else:
#                 reward[tt,ii] = stochastic_env(R,ii) 
#                 N_tot[ii] = N_tot[ii] + 1
                    
#         N = [N_tot[i] for i in range(n_arms) if A[tt,i]==1]
            
#     rewards = A*R-max(R)
#     opt_arm = np.argmax(R) 
#     regret = [0]*(T+1)
#     for i in range(T):
#         for j in range(n_arms):
#             if j!=opt_arm and rewards[i,j]==-max(R):
#                 rewards[i,j] = 0
#             if j==opt_arm and rewards[i,j]==0:
#                 rewards[i,opt_arm] = max(R)           
#         regret = i*max(R) - np.sum(rewards[n_arms:])

#     return regret,np.sum(A,0)
       
# def UCBrev_run(T,n_arms,R):
    
#     from stochastic_env import stochastic_env
#     def n_m(deltam,rounds):
#         n_m = np.ceil(2*np.log(rounds*deltam**2)/(deltam**2))
#         return n_m
    
#     deltam = 1
#     N = np.zeros((T,n_arms),float)
#     cum_r = np.zeros((T,n_arms),float)
    
#     ind = [0] * T
#     rewards = [0] * T 
#     arms_in_use = [1] * n_arms
#     arms_in_use_idex = [i for i in range(n_arms)]

    
#     for tt in range(1,T):
        
#         N_used_arms = [sum(N[:,i]) for i in range(n_arms) if arms_in_use[i]==1]
#         arms_in_use_index = [i for i,x in enumerate(arms_in_use) if x == 1]

#         if all(i==n_m(deltam,T) for i in N_used_arms):
#             reward_used_arms = cum_r[0:tt,arms_in_use_index]
#             avg_reward_used_arms = np.mean(reward_used_arms,0)
#             print(tt)
#             bound = np.sqrt(np.log(T*deltam**2)/(2*n_m(deltam,T)))
#             deleted_arms = avg_reward_used_arms+bound<max(avg_reward_used_arms)-bound
#             print(avg_reward_used_arms+bound)
#             print(max(avg_reward_used_arms)-bound)
#             print(avg_reward_used_arms+bound<max(avg_reward_used_arms)-bound)
#             arms_in_use = [int(not i) for i in deleted_arms]  
#             deltam = deltam/2
        
#         ind_temp = tt%sum(arms_in_use)
#         ind[tt] = arms_in_use_index[ind_temp]
#         # print('tt :',tt)
#         # print('ind: ', ind[tt])
#         #print(arms_in_use)
#         rewards[tt] = stochastic_env(R, ind[tt]) 
#         N[tt,ind[tt]] = 1
#         cum_r[tt,ind[tt]] = rewards[tt]
    
#     ## Final regret estimate
#     #get rewards of played arms. -2*max(R) is due to the fact that R[i]-1*max(R) is 
#     #the difference between expected rewards and the additional -max(R) is due to the 
#     #fact that the optimal arm is not pulled
#     rewards = [R[i]-2*max(R) for i in ind if i is not None] 
#     #correct reward if optimal arm is pulled
#     rewards = [max(R) if i == -max(R) else i for i in rewards] 
#     regret = (T*max(R)-sum(rewards[1:])) #First index removed as not used
    
#     return regret, sum(sum(N))