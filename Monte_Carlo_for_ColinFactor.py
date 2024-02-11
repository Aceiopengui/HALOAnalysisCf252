import numpy as np
import matplotlib.pyplot as plt
import random as random


# =============================================================================
# rate=1
# length=3600
# timeWindow=0.002
# =============================================================================
#Takes rate in neutrons per second, length in seconds, time window in seconds
def MultipleFactorComputer(rate,length,timeWindow):
    timestamps=[]
    bursts=[]
    j=0
    oldtime=0
    numN_event=0
    totalEventN=rate*length
    for i in range(int(totalEventN)):
        timestamps.append(random.uniform(0,length))
    
    timestamps=sorted(timestamps,key=float)
    
    
    while j<len(timestamps):
        currenttime=timestamps[j]
        if(currenttime-oldtime<timeWindow):
            #New event in burst
            numN_event=numN_event+1
            #oldtime=currenttime
        else:
            bursts.append(numN_event)
            numN_event=1
        oldtime=currenttime
        j=j+1
        
        
        
    #print(bursts,"bursts")    
    #print(len(timestamps),"How many events")
    #print(len(bursts))
    count_1=bursts.count(1)
    count_2=bursts.count(2)
    count_3=bursts.count(3)
    count_4=bursts.count(4)
    count_5=bursts.count(5)
   # print(count_1,"1")
   # print(count_2,"2")
   # print(count_3,"3")
  #  print(count_4,"4")
   # print(count_5,"5")
    prop_single=count_1/len(bursts)
    prop_double=count_2/(len(bursts))
    prop_triple=count_3/(len(bursts))
    return prop_single,prop_double,prop_triple
    
    
    




points=5
fissionRate=np.linspace(1,points*10,100)

timeWindow=0.002
colinFactorSingleOld = np.exp(-2*timeWindow*fissionRate)
colinFactorDoubleOld = 3*np.exp(-fissionRate*timeWindow)-2*np.exp(-2*fissionRate*timeWindow)-1
colinFactorTripleOld = (1-np.exp(-fissionRate*timeWindow))**2
        #New Ones below
#if __name__=='__main__':
i=1
single_array=[]
double_array=[]
triple_array=[]
xarray=[]
while i<=points:    
    prop_single,prop_double,prop_triple=MultipleFactorComputer(10*i,14402,0.002)
    single_array.append(prop_single)
    double_array.append(prop_double)
    triple_array.append(prop_triple)
    xarray.append(i*10)
    i=i+1
    
preNormColinFactorZero=np.exp(-fissionRate*timeWindow)
preNormColinFactorSingle = np.exp(-2*fissionRate*timeWindow)*(np.exp(fissionRate*timeWindow)-1)
preNormColinFactorDouble = np.exp(-3*fissionRate*timeWindow)*(np.exp(fissionRate*timeWindow)-1)**2
preNormColinFactorTriple = np.exp(-4*fissionRate*timeWindow)*(np.exp(fissionRate*timeWindow)-1)**3
colinFactorSum=preNormColinFactorSingle+preNormColinFactorDouble+preNormColinFactorTriple
colinFactorSingle=preNormColinFactorSingle/colinFactorSum
colinFactorDouble=preNormColinFactorDouble/colinFactorSum
colinFactorTriple=preNormColinFactorTriple/colinFactorSum


fig,ax=plt.subplots(dpi=500)
fig2,ax2=plt.subplots(dpi=500)
fig3,ax3=plt.subplots(dpi=500)
ax.plot(fissionRate,colinFactorSingleOld,label="Old Method Probability of Single Fission")
ax.plot(fissionRate,colinFactorSingle,label="New Method Probability of Single Fission")
ax.scatter(xarray,single_array,label="Probability of Single Fission given by Monte Carlo simulation",c='g')
ax.set_xlabel("Fissions Per Second")
ax.set_ylabel("Likelihood of Detection being One Fission")
ax.set_title("Single Fission Detection Probability")
ax.legend(prop={'size': 6})
ax2.plot(fissionRate,colinFactorDoubleOld,label="Old Method Probability of Double Fission")
ax2.plot(fissionRate,colinFactorDouble,label="New Method Probability of Double Fission")
ax2.scatter(xarray,double_array,label="Probability of Double Fission given by Monte Carlo simulation",c='g')
ax2.set_xlabel("Fissions Per Second")
ax2.set_ylabel("Likelihood of Detection being Two Fissions")
ax2.set_title("Double Fission Detection Probability")
ax2.legend(prop={'size': 6})
ax3.plot(fissionRate,colinFactorTripleOld,label="Old Method Probability of Triple Fission")
ax3.plot(fissionRate,colinFactorTriple,label="New Method Probability of Triple Fission")
ax3.scatter(xarray,triple_array,label="Probability of Triple Fission given by Monte Carlo simulation",c='g')
ax3.set_xlabel("Fissions Per Second")
ax3.set_ylabel("Likelihood of Detection being Three Fissions")
ax3.set_title("Triple Fission Detection Probability")
ax3.legend(prop={'size': 6})


plt.show()




print(prop_single,"Proportion of counts that are single")
print(prop_double,"prop double")
print(prop_triple,"prop triple")

print(prop_single+prop_double+prop_triple,"sum")
print(1-(prop_single+prop_double+prop_triple),"Probability of 4 and above counts")
