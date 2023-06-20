import numpy as np

def stochastic_env(R,a):
    n_arms = len(R)
    all_r = [np.random.random() for i in range(n_arms)]
    r_tot = [all_r[i] < R[i] for i in range(n_arms)]
    
    r = np.take(r_tot,a)
      
    return r