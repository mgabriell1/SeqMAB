import os
# Set working directory same as file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import sys
sys.path.append(dname+'/Algorithms')
sys.path.append(dname+'/Environments')
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle

from SEQUCBE import SEQUCBE_run, SEQUCBE_NoLim_run, SEQUCBE
from UCBE import UCBE_run, UCBE
from SR import SR_run

# Random problem generation details
expName = 'Audibertea2010'
saveData = False

# Parameter c value
c = 1

Rp = 7
R_Rp = [[0.5] for _ in range(Rp)]
R_Rp[0] = R_Rp[0] + [0.4]*19
R_Rp[1] = R_Rp[1] + [0.42]*5 + [0.38]*14
R_Rp[2] = R_Rp[2] + [0.5-(0.37)**i for i in [2,3,4]]
R_Rp[3] = R_Rp[3] + [0.42,0.4,0.4,0.35,0.35]
R_Rp[4] = R_Rp[4] + [0.5-0.025*i for i in range(2,16)]
R_Rp[5] = R_Rp[5] + [0.48] + [0.37]*18
R_Rp[6] = R_Rp[6] + [0.45]*5 + [0.43]*14 + [0.38]*10

tau_Rp = [2000, 2000, 2000, 600, 4000, 6000, 6000]

# Number of tries for every random problem to estimate error percentage
Tries = 100

# Number of iterations to estimate error percentage standard deviation
Reps = 10

# Number of algorithms to be compared
Algs = 4
AlgsNames=['SEQ(UCBE)-LR','SEQ(UCBE)-LP','UCBE','SR+']

errorPercSEQUCBE_LR = [[None]*Reps for _ in range(Rp)]
errorPercSEQUCBE_LP = [[None]*Reps for _ in range(Rp)]
errorPercUCBE = [[None]*Reps for _ in range(Rp)]
errorPercSR = [[None]*Reps for _ in range(Rp)]

for kk in range(Reps):
    
    # Create array to store results
    bestArmSEQUCBE_LR = [[0]*Tries for _ in range(Rp)]
    bestArmSEQUCBE_LP = [[0]*Tries for _ in range(Rp)]
    bestArmUCBE = [[0]*Tries for _ in range(Rp)]
    bestArmSR = [[0]*Tries for _ in range(Rp)]
    
    NPullsSEQUCBE_LR = [[0]*Tries for _ in range(Rp)]
    NPullsSEQUCBE_LP = [[0]*Tries for _ in range(Rp)]
    NPullsUCBE = [[0]*Tries for _ in range(Rp)]
    NPullsSR = [[0]*Tries for _ in range(Rp)]
    
    indSEQUCBE_LR = [[0]*Tries for _ in range(Rp)]
    indSEQUCBE_LP = [[0]*Tries for _ in range(Rp)]
    indUCBE = [[0]*Tries for _ in range(Rp)]
    indSR = [[0]*Tries for _ in range(Rp)]
    
    finalTimeSEQUCBE_LP = [[0]*Tries for _ in range(Rp)]

    for ii in range(Rp):
        # Create random problem
        #R = rng.random(size=n_arms)
        R =  np.array(R_Rp[ii])
        tau = tau_Rp[ii]
        n_arms = len(R)
        
        for tr in range(Tries):
            print('Rep: ', kk, ', R_p: ',ii,', try: ', tr)
            
            # SEQ-UCBE LR             
            bestArmSEQUCBE_LR[ii][tr], NPullsSEQUCBE_LR[ii][tr], indSEQUCBE_LR[ii][tr] = SEQUCBE_NoLim_run(tau,n_arms,R,c)
            
            # SEQ-UCBE LP
            bestArmSEQUCBE_LP[ii][tr], NPullsSEQUCBE_LP[ii][tr], indSEQUCBE_LP[ii][tr], finalTimeSEQUCBE_LP[ii][tr] = SEQUCBE_run(tau,n_arms,R,c,tau)
        
            # UCBE
            bestArmUCBE[ii][tr], NPullsUCBE[ii][tr], indUCBE[ii][tr] = UCBE_run(tau,n_arms,R,c)
            
            # SR
            bestArmSR[ii][tr], NPullsSR[ii][tr], indSR[ii][tr] = SR_run(tau,n_arms,R)
    
        
        
    ##
    bestArmTotal = []
    bestArmTotal.append(bestArmSEQUCBE_LR)
    bestArmTotal.append(bestArmSEQUCBE_LP)
    bestArmTotal.append(bestArmUCBE)
    bestArmTotal.append(bestArmSR)
    
    NPullsTotal = []
    NPullsTotal.append(NPullsSEQUCBE_LR)
    NPullsTotal.append(NPullsSEQUCBE_LP)
    NPullsTotal.append(NPullsUCBE)
    NPullsTotal.append(NPullsSR)
    
    indTotal = []
    indTotal.append(indSEQUCBE_LR)
    indTotal.append(indSEQUCBE_LP)
    indTotal.append(indUCBE)
    indTotal.append(indSR)
                
    
    for r in range(Rp):
        R = R_Rp[r]
        
        arm_star = np.argmax(R)
        # errorPercSEQUCBE_Rp = []
        # errorPercUCBE_Rp = []
        # errorPercSR_Rp = []
        
        correctSEQUCBE_LR = [i == arm_star for i in bestArmSEQUCBE_LR[r]]
        errorPercSEQUCBE_LR[r][kk] = (1 - sum(correctSEQUCBE_LR)/len(correctSEQUCBE_LR))
        
        correctSEQUCBE_LP = [i == arm_star for i in bestArmSEQUCBE_LP[r]]
        errorPercSEQUCBE_LP[r][kk] = (1 - sum(correctSEQUCBE_LP)/len(correctSEQUCBE_LP))
        
        correctUCBE = [i == arm_star for i in bestArmUCBE[r]]
        errorPercUCBE[r][kk] = (1 - sum(correctUCBE)/len(correctUCBE))
        
        correctSR = [i == arm_star for i in bestArmSR[r]]
        errorPercSR[r][kk] = (1 - sum(correctSR)/len(correctSR))
  
##
errorPercSEQUCBE_LR_Avg = np.mean(errorPercSEQUCBE_LR,1)
errorPercSEQUCBE_LR_Sd = np.std(errorPercSEQUCBE_LR,1)*1.96/np.sqrt(Reps)

errorPercSEQUCBE_LP_Avg = np.mean(errorPercSEQUCBE_LP,1)
errorPercSEQUCBE_LP_Sd = np.std(errorPercSEQUCBE_LP,1)*1.96/np.sqrt(Reps)

errorPercUCBE_Avg = np.mean(errorPercUCBE,1)
errorPercUCBE_Sd = np.std(errorPercUCBE,1)*1.96/np.sqrt(Reps)

errorPercSR_Avg = np.mean(errorPercSR,1)
errorPercSR_Sd = np.std(errorPercSR,1)*1.96/np.sqrt(Reps)

x = np.arange(Rp)  
width = 0.2
fig, axs = plt.subplots(1, 1, figsize=(9, 6))
axs.bar(x - 1.5*width,errorPercSEQUCBE_LR_Avg,width,label=AlgsNames[0])
axs.errorbar(x - 1.5*width,errorPercSEQUCBE_LR_Avg,yerr=errorPercSEQUCBE_LR_Sd,c="black",capsize=5,fmt='none')
axs.bar(x - 0.5*width,errorPercSEQUCBE_LP_Avg,width,label=AlgsNames[1])
axs.errorbar(x - 0.5*width,errorPercSEQUCBE_LP_Avg,yerr=errorPercSEQUCBE_LP_Sd,c="black",capsize=5,fmt='none')
axs.bar(x + 0.5*width,errorPercUCBE_Avg,width,label=AlgsNames[2])
axs.errorbar(x + 0.5*width,errorPercUCBE_Avg,yerr=errorPercUCBE_Sd,c="black",capsize=5,fmt='none')
axs.bar(x + 1.5*width,errorPercSR_Avg,width,label=AlgsNames[3])
axs.errorbar(x + 1.5*width,errorPercSR_Avg,yerr=errorPercSR_Sd,c="black",capsize=5,fmt='none')
axs.set(ylabel='Probability of error [-]',xlabel='Experiment')
axs.legend()
if saveData:
    plt.savefig('ErrorProb_BAI_comparison_'+expName+'.png', dpi=600,bbox_inches = 'tight')
     
if saveData:
    with open('bestArmTotal_BAI_'+expName+'_rep'+str(kk)+'.pickle', 'wb') as f:
        pickle.dump(bestArmTotal, f, pickle.HIGHEST_PROTOCOL)
        
    with open('NPullsTotal_BAI_'+expName+'_rep'+str(kk)+'.pickle', 'wb') as f:
        pickle.dump(NPullsTotal, f, pickle.HIGHEST_PROTOCOL)
        
    with open('indTotal_BAI_'+expName+'_rep'+str(kk)+'.pickle', 'wb') as f:
        pickle.dump(indTotal, f, pickle.HIGHEST_PROTOCOL)
        
    with open('finalTimeSEQUCBE_LP_BAI_'+expName+'_rep'+str(kk)+'.pickle', 'wb') as f:
        pickle.dump(finalTimeSEQUCBE_LP, f, pickle.HIGHEST_PROTOCOL)
  

