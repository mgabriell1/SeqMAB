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

from stochastic_env import stochastic_env 
from SEQUCB import SEQUCB1_run, SEQUCB1
from UCB1 import UCB1_run, UCB1
from UCBrev import UCBrev_run
from SEQTS import SEQTS_run, SEQTS
from TS import TS_run, TS

# Random problem generation details
tau = 1000
n_arms = 10 
Delta = 0.1 # Set as None to not set Delta_min
Rdistr = 'rand' #fixed/rand. 'fixed' sets a constant delta between all arms, while 'rand' only assures that Delta_min is guaranteed
expName = '10arms_delta0.1_rand'
saveData = False

rng = np.random.default_rng()

# Number of random problems
Rp = 1

# Number of tries for every random problem
Tries = 3

# Create folder to save files
if saveData:
    if not os.path.exists(expName):
        os.makedirs(expName)
        
# Number of algorithms to be compared
Algs = 5
AlgsNames=['SEQ(UCB1)','UCB1','UCBrev+','SEQ(TS)','TS']

# Create array to store results
RegretSEQUCB = [[0]*Tries for _ in range(Rp)]
RegretUCB1 = [[0]*Tries for _ in range(Rp)]
RegretUCBrev = [[0]*Tries for _ in range(Rp)]
RegretSEQTS = [[0]*Tries for _ in range(Rp)]
RegretTS = [[0]*Tries for _ in range(Rp)]

NPullsSEQUCB = [[0]*Tries for _ in range(Rp)]
NPullsUCB1 = [[0]*Tries for _ in range(Rp)]
NPullsUCBrev = [[0]*Tries for _ in range(Rp)]
NPullsSEQTS = [[0]*Tries for _ in range(Rp)]
NPullsTS = [[0]*Tries for _ in range(Rp)]

indSEQUCB = [[0]*Tries for _ in range(Rp)]
indUCB1 = [[0]*Tries for _ in range(Rp)]
indUCBrev = [[0]*Tries for _ in range(Rp)]
indSEQTS = [[0]*Tries for _ in range(Rp)]
indTS = [[0]*Tries for _ in range(Rp)]

NpullsRoundUCBrev = [[0]*Tries for _ in range(Rp)]

R_Rp = [0]*Rp


for ii in range(Rp):
    # Create random problem
    if Delta != None:
        if Rdistr == 'fixed':
            R = rng.random()*(1-n_arms*Delta)+n_arms*Delta
            R = np.append(np.array(R),[R-i*Delta for i in range(1,n_arms)])
        else:
            R = rng.random(size=(n_arms-1))*(1-Delta)
            R = np.append(np.array(max(R)+Delta),R)
            
    else:
        R = rng.random(size=n_arms)
    
    R_Rp[ii] = R
    
    for tr in range(Tries):
        print('R_p: ',ii,', try: ', tr)
        
        # SEQ-UCB
        RegretSEQUCB[ii][tr], NPullsSEQUCB[ii][tr], indSEQUCB[ii][tr] = SEQUCB1_run(tau,n_arms,R)
    
        # UCB1
        RegretUCB1[ii][tr], NPullsUCB1[ii][tr], indUCB1[ii][tr] = UCB1_run(tau,n_arms,R)
                
        # UCBrev
        RegretUCBrev[ii][tr], NPullsUCBrev[ii][tr], NpullsRoundUCBrev[ii][tr], indUCBrev[ii][tr] = UCBrev_run(tau,n_arms,R)
       
        #SEQ-TS
        RegretSEQTS[ii][tr], NPullsSEQTS[ii][tr], indSEQTS[ii][tr] = SEQTS_run(tau,n_arms,R)        
    
        # TS
        RegretTS[ii][tr], NPullsTS[ii][tr], indTS[ii][tr] = TS_run(tau,n_arms,R)

        
    # Results plot (at the end of each Rp)
    RegretRp = []
    RegretRp.append(RegretSEQUCB[ii])
    RegretRp.append(RegretUCB1[ii])
    RegretRp.append(RegretUCBrev[ii])
    RegretRp.append(RegretSEQTS[ii])
    RegretRp.append(RegretTS[ii])
    RegretRp_Avg = []
    RegretRp_Std = []
    for zz in range(Algs):
        RegretRp_Avg.append(np.mean(RegretRp[zz],0))
        RegretRp_Std.append(np.std(RegretRp[zz],0)*1.96/np.sqrt(Tries))
    fig, axs = plt.subplots(1, 1, figsize=(9, 6))
    for zz in range(Algs):
        axs.plot(range(tau+1),RegretRp_Avg[zz],label=AlgsNames[zz])
        axs.fill_between(range(tau+1),y1=RegretRp_Avg[zz]+RegretRp_Std[zz],y2=RegretRp_Avg[zz]-RegretRp_Std[zz],alpha=0.25)
    axs.set(xlabel='Rounds [-]', ylabel='Empirical pseudo-regret [-]')
    plt.title('Random problem '+str(ii))
    axs.legend()
    if saveData:
        plt.savefig(expName+'/Regret_comparison_'+expName+'_Rp_'+str(ii)+'.png', dpi=600,bbox_inches = 'tight')
    
    #
    best_arm = np.argmax(R)
    
    SEQUCB_Rp_pulled_arms = [[i for i in j if i!=None] for j in  indSEQUCB[ii]]
    SEQUCB_Rp_correct_pulls = [[i == best_arm for i in j] for j in  SEQUCB_Rp_pulled_arms]
    SEQUCB_Rp_perc_correct_pulls = [sum(i)/len(i) for i in  SEQUCB_Rp_correct_pulls]
    SEQUCB_Rp_rounds_with_correct_pulls = [sum(i)/tau for i in  SEQUCB_Rp_correct_pulls]
    SEQUCB_Rp_CP_avg = np.mean(SEQUCB_Rp_perc_correct_pulls)
    SEQUCB_Rp_CP_std = np.std(SEQUCB_Rp_perc_correct_pulls)*1.96/np.sqrt(Tries)
    SEQUCB_Rp_RCP_avg = np.mean(SEQUCB_Rp_rounds_with_correct_pulls)
    SEQUCB_Rp_RCP_std = np.std(SEQUCB_Rp_rounds_with_correct_pulls)*1.96/np.sqrt(Tries)
    
    UCB_Rp_pulled_arms = [[i for i in j if i!=None] for j in  indUCB1[ii]]
    UCB_Rp_correct_pulls = [[i == best_arm for i in j] for j in  UCB_Rp_pulled_arms]
    UCB_Rp_perc_correct_pulls = [sum(i)/len(i) for i in  UCB_Rp_correct_pulls]
    UCB_Rp_rounds_with_correct_pulls = [sum(i)/tau for i in  UCB_Rp_correct_pulls]
    UCB_Rp_CP_avg = np.mean(UCB_Rp_perc_correct_pulls)
    UCB_Rp_CP_std = np.std(UCB_Rp_perc_correct_pulls)*1.96/np.sqrt(Tries)
    UCB_Rp_RCP_avg = np.mean(UCB_Rp_rounds_with_correct_pulls)
    UCB_Rp_RCP_std = np.std(UCB_Rp_rounds_with_correct_pulls)*1.96/np.sqrt(Tries)
    
    SEQTS_Rp_pulled_arms = [[i for i in j if i!=None] for j in  indSEQTS[ii]]
    SEQTS_Rp_correct_pulls = [[i == best_arm for i in j] for j in  SEQTS_Rp_pulled_arms]
    SEQTS_Rp_perc_correct_pulls = [sum(i)/len(i) for i in  SEQTS_Rp_correct_pulls]
    SEQTS_Rp_rounds_with_correct_pulls = [sum(i)/tau for i in  SEQTS_Rp_correct_pulls]
    SEQTS_Rp_CP_avg = np.mean(SEQTS_Rp_perc_correct_pulls)
    SEQTS_Rp_CP_std = np.std(SEQTS_Rp_perc_correct_pulls)*1.96/np.sqrt(Tries)
    SEQTS_Rp_RCP_avg = np.mean(SEQTS_Rp_rounds_with_correct_pulls)
    SEQTS_Rp_RCP_std = np.std(SEQTS_Rp_rounds_with_correct_pulls)*1.96/np.sqrt(Tries)
    
    TS_Rp_pulled_arms = [[i for i in j if i!=None] for j in  indTS[ii]]
    TS_Rp_correct_pulls = [[i == best_arm for i in j] for j in  TS_Rp_pulled_arms]
    TS_Rp_perc_correct_pulls = [sum(i)/len(i) for i in  TS_Rp_correct_pulls]
    TS_Rp_rounds_with_correct_pulls = [sum(i)/tau for i in  TS_Rp_correct_pulls]
    TS_Rp_CP_avg = np.mean(TS_Rp_perc_correct_pulls)
    TS_Rp_CP_std = np.std(TS_Rp_perc_correct_pulls)*1.96/np.sqrt(Tries)
    TS_Rp_RCP_avg = np.mean(TS_Rp_rounds_with_correct_pulls)
    TS_Rp_RCP_std = np.std(TS_Rp_rounds_with_correct_pulls)*1.96/np.sqrt(Tries)
    
    UCBrev_Rp_pulled_arms = [[i for i in j if i!=None] for j in  indUCBrev[ii]]
    UCBrev_Rp_correct_pulls = [[i == best_arm for i in j] for j in  UCBrev_Rp_pulled_arms]
    UCBrev_Rp_perc_correct_pulls = [sum(i)/len(i) for i in  UCBrev_Rp_correct_pulls]
    UCBrev_Rp_rounds_with_correct_pulls = [sum(i)/tau for i in  UCBrev_Rp_correct_pulls]
    UCBrev_Rp_CP_avg = np.mean(UCBrev_Rp_perc_correct_pulls)
    UCBrev_Rp_CP_std = np.std(UCBrev_Rp_perc_correct_pulls)*1.96/np.sqrt(Tries)
    UCBrev_Rp_RCP_avg = np.mean(UCBrev_Rp_rounds_with_correct_pulls)
    UCBrev_Rp_RCP_std = np.std(UCBrev_Rp_rounds_with_correct_pulls)*1.96/np.sqrt(Tries)
    
    fig = plt.figure(figsize=(7,5))
    plt.bar(0,SEQUCB_Rp_CP_avg,yerr=SEQUCB_Rp_CP_std,capsize=5,zorder=1)
    plt.scatter(0,SEQUCB_Rp_RCP_avg,c='black',zorder=2)
    plt.errorbar(0,SEQUCB_Rp_RCP_avg,SEQUCB_Rp_RCP_std,capsize=5,c='black',zorder=2)
    plt.bar(1,UCB_Rp_CP_avg,yerr=UCB_Rp_CP_std,capsize=5,zorder=1)
    plt.scatter(1,UCB_Rp_RCP_avg,c='black',zorder=2)
    plt.errorbar(1,UCB_Rp_RCP_avg,UCB_Rp_RCP_std,capsize=5,c='black',zorder=2)
    plt.bar(2,UCBrev_Rp_CP_avg,yerr=UCBrev_Rp_CP_std,capsize=5,zorder=1)
    plt.scatter(2,UCBrev_Rp_RCP_avg,c='black',zorder=2)
    plt.errorbar(2,UCBrev_Rp_RCP_avg,UCBrev_Rp_RCP_std,capsize=5,c='black',zorder=2)
    plt.bar(3,SEQTS_Rp_CP_avg,yerr=SEQTS_Rp_CP_std,capsize=5,zorder=1)
    plt.scatter(3,SEQTS_Rp_RCP_avg,c='black',zorder=2)
    plt.errorbar(3,SEQTS_Rp_RCP_avg,SEQTS_Rp_RCP_std,capsize=5,c='black',zorder=2)
    plt.bar(4,TS_Rp_CP_avg,yerr=TS_Rp_CP_std,capsize=5,zorder=1)
    plt.scatter(4,TS_Rp_RCP_avg,c='black',zorder=2)
    plt.errorbar(4,TS_Rp_RCP_avg,TS_Rp_RCP_std,capsize=5,c='black',zorder=2)
    plt.xticks(np.arange(0,len(AlgsNames)),AlgsNames,rotation=45)
    plt.ylabel('Percentage [-]')
    if saveData:
        plt.savefig(expName+'/Percentage_comparison_'+expName+'_Rp_'+str(ii)+'.png', dpi=600,bbox_inches = 'tight')
            

##
RegretTotal = []
RegretTotal.append(RegretSEQUCB)
RegretTotal.append(RegretUCB1)
RegretTotal.append(RegretUCBrev)
RegretTotal.append(RegretSEQTS)
RegretTotal.append(RegretTS)


NPullsTotal = []
NPullsTotal.append(NPullsSEQUCB)
NPullsTotal.append(NPullsUCB1)
NPullsTotal.append(NPullsUCBrev)
NPullsTotal.append(NPullsSEQTS)
NPullsTotal.append(NPullsTS)

indTotal = []
indTotal.append(indSEQUCB)
indTotal.append(indUCB1)
indTotal.append(indUCBrev)
indTotal.append(indSEQTS)
indTotal.append(indTS)



if saveData:
    with open(expName+'/RegretTotal_'+expName+'.pickle', 'wb') as f:
        pickle.dump(RegretTotal, f, pickle.HIGHEST_PROTOCOL)
        
    with open(expName+'/NPullsTotal_'+expName+'.pickle', 'wb') as f:
        pickle.dump(NPullsTotal, f, pickle.HIGHEST_PROTOCOL)
        
    with open(expName+'/indTotal_'+expName+'.pickle', 'wb') as f:
        pickle.dump(indTotal, f, pickle.HIGHEST_PROTOCOL)
        
    with open(expName+'/R_Rp_'+expName+'.pickle', 'wb') as f:
        pickle.dump(R_Rp, f, pickle.HIGHEST_PROTOCOL)

    with open(expName+'/NpullsRoundUCBrev_'+expName+'.pickle', 'wb') as f:
        pickle.dump(NpullsRoundUCBrev, f, pickle.HIGHEST_PROTOCOL)


