import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from itertools import zip_longest
import scipy.optimize
import numdifftools
#np.set_printoptions(suppress=True)
import csv
import pandas as pd
from scipy.optimize import curve_fit
import random as random
from Monte_Carlo_for_ColinFactor import MultipleFactorComputer
#TO DO TOMORROW: pinpoint where the discontinuity in the amount takes place by computing L in a loop for a single case here
#similar to before and make a graph of the parabola similar to before

#DataGenerator takes in a list of years, # of fissions at time 0, initial Proportions of Cf252 and Cf250 by mass at t0, a list of each of the runlengths
def DataGenerator(Years,Fissions,Eff,Cf252Prop,Cf250Prop,MysteryIsotope,RunLength,BackgroundRate,PrintResults,GenerateTotalCounts,JustFissions=False):
    #Defining some natural parameters
    decayConst252,decayConst250,decayConst248=np.log(2)/2.65,np.log(2)/13.08,np.log(2)/348000
    chanAlpha252,chanAlpha250,chanAlpha248=0.96908,0.99923,0.9161
    chanFiss252,chanFiss250,chanFiss248=1-chanAlpha252,1-chanAlpha250,1-chanAlpha248
    #Time window for which two detections need to be apart to be considered separate fissions
    timeWindow=0.002
    #Proportion of each spontaneous fission that yields n neutrons (index starting at 1) Continues out to account for multiple counts
    cf252MultiNoNorm=[0.026,0.1267,0.2734,0.3039,0.1848,0.0657,0.0154,0.0020,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    cf250MultiNoNorm=[0.03594,0.16771,0.29635,0.30104,0.14531,0.04740,0.00417,0.00208,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    cm248MultiNoNorm=[0.05986,0.22313,0.34966,0.25306,0.09048,0.01769,0.00340,0.00068,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    ####
    cm246MultiNoNorm=[0.076,0.264,0.347,0.218,0.074,0.005,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    pu240MultiNoNorm=[0.2325,0.335,0.25375,0.0975,0.016875,0.001245,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    pu242MultiNoNorm=[0.22807,0.3334,0.246603,0.09876,0.01696,0.00122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ####
    mult252=3.735
    mult250=3.50
    
    #Sets the value for the neutron rate from background neutrons to be 1 per second in the 1 bin
    #backgroundRate=[0.0671]
    listOfNeutrons=[]
    #Above distributions normalized
    Cf252Multi=[x / sum(cf252MultiNoNorm) for x in cf252MultiNoNorm]
    Cf250Multi=[x / sum(cf250MultiNoNorm) for x in cf250MultiNoNorm]
    Cm248Multi=[x / sum(cm248MultiNoNorm) for x in cm248MultiNoNorm]
    Cm246Multi=[x / sum(cm246MultiNoNorm) for x in cm246MultiNoNorm]
    Pu240Multi=[x / sum(pu240MultiNoNorm) for x in pu240MultiNoNorm]
    Pu242Multi=[x / sum(pu242MultiNoNorm) for x in pu242MultiNoNorm]
    pushToDateFitter=[]
    fissionsForGraph=[]
    datesForGraph=[]
    #Run this function once for each element in the "years" list passed in, generating a set of data corresponding to a run taken in that year
    for n in range(len(Years)):
        #This if statement sets up a factor that ensures that runs of different lengths are handled properly
        if(len(Years)==1):
            runLengthFactor=1
        else:
            runLengthFactor=RunLength[n]/RunLength[0]
            
            
        #MysteryMultList=[m1,m2,m3,m4,m5,m6,m7,m8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #test1q=[round(item,4) for item in Cf252Multi]
        #print(test1q)
        
        
        
        
        #Determines the total number of background events expected (Not including detector efficiency)
        #backGroundEvents=[x*RunLength[n] for x in BackgroundRate[n]]
        backGroundEvents=RunLength[n]*BackgroundRate[n]
        #Determining the initial proportions of fissions that each isotope is responsible for at t0
       # initialFissionProportion252,initialFissionProportion250,initialFissionProportion248=FissionProportions(Cf252Prop,Cf250Prop,Cm248Prop,decayConst252,decayConst250,decayConst248,chanFiss252,chanFiss250,chanFiss248)
        initialFissionProportion252,initialFissionProportion250=FissionProportions(Cf252Prop,Cf250Prop,decayConst252,decayConst250,chanFiss252,chanFiss250)
        mass248AfterInf=(((initialFissionProportion252*Fissions*mult252)/(decayConst252*chanFiss252*mult252))*chanAlpha252)*chanFiss248*3.15*decayConst248
        Q=(initialFissionProportion252*Fissions*mult252)*np.exp(-decayConst252*Years[n])+(initialFissionProportion250*Fissions*mult250)*np.exp(-decayConst250*Years[n])+(mass248AfterInf*(1-np.exp(-decayConst252*Years[n])))
       # print(Q,"Q")
        #MysteryIsotopeFissions=[i*MysteryIsotope/mult250*RunLength[n] for i in Cf250Multi]
        MysteryIsotopeFissions=RunLength[n]*MysteryIsotope/mult252
        #print(MysteryIsotope,MysteryIsotopeFissions,"mysteries")
        #print(MysteryIsotope*RunLength[n])
        #mysteryIsotope=Cm248Prop*Fissions
        #print(Fissions,"Fisisons")
        initialFissions252,initialFissions250=InitialFissionComputer(initialFissionProportion252,Fissions),InitialFissionComputer(initialFissionProportion250,Fissions)#,InitialFissionComputer(initialFissionProportion248,Fissions)
       # print(initialFissions252,"Initial Fisisons",Years[n])
        #Ages the above "initialFissions" variables to the current time, giving the amount of fissions from each at time "Years[n]"
        currentFissions252,currentFissions250=FissionComputer(initialFissions252,decayConst252,Years[n],runLengthFactor),FissionComputer(initialFissions250,decayConst250,Years[n],runLengthFactor),#FissionComputer(initialFissions248,decayConst248,Years[n],runLengthFactor)
        #print(currentFissions252,currentFissions250,"Current Fissions",Years[n])
        #Computes the amount of fissions from Cm248 by first converting the amount of fissions from Cf252 into arbitrary mass units, then gets the amount that would have decayed and the percent that would decay to curium, then converts that mass of curium into fisisons of Cm248
        currentFissions248=FissionComputer248(initialFissions252,currentFissions252,decayConst252,decayConst248,chanFiss252,chanAlpha252,chanFiss248)#,currentFissionsInitial248)
        #Gets the total fissions at the time of interest by adding contributions from the 3 isotopes
        totalCurrentFissions=currentFissions252+currentFissions250+currentFissions248+MysteryIsotopeFissions
        #print(currentFissions252,currentFissions250,currentFissions248,MysteryIsotopeFissions,"fissions test")
        #Gets the total rate of fissions at our times of interest, since we know the runlength and the total number of fissions and we are expecting the rate to be constant within one run to very high precission
        fissionRate=FissionRate(totalCurrentFissions,RunLength[n])
        if JustFissions:
            fissionsForGraph.append((totalCurrentFissions/RunLength[n]))
            #print(fissionsForGraph,"Fissionsforgraph")

            if Years[n]==Years[-1]:
                return fissionsForGraph
            else:
                continue
            
        #print(Cf250MassFractionGrabber(currentFissions252,currentFissions250,decayConst252,decayConst250,chanFiss252,chanFiss250,mult252,mult250),"Mass fraction of 250 at given time (should increase)")

        preEffDistTotal=DistComputer(currentFissions252,currentFissions250,currentFissions248,MysteryIsotopeFissions,Cf252Multi,Cf250Multi,Cm248Multi)
        #Gets factors pertaining to multiple counts outlined in page 42 of Colin's thesis. TO DO: Reserarch where these come form
        
        colinFactorSingleOld = np.exp(-2*totalCurrentFissions*timeWindow/RunLength[n])
        colinFactorDoubleOld = 3*np.exp(-totalCurrentFissions*timeWindow/RunLength[n])-2*np.exp(-2*totalCurrentFissions*timeWindow/RunLength[n])-1
        colinFactorTripleOld = (1-np.exp(-totalCurrentFissions*timeWindow/RunLength[n]))**2
        #New Ones below
        preNormColinFactorSingle = np.exp(-2*fissionRate*timeWindow)*(np.exp(fissionRate*timeWindow)-1)
        preNormColinFactorDouble = np.exp(-3*fissionRate*timeWindow)*(np.exp(fissionRate*timeWindow)-1)**2
        preNormColinFactorTriple = np.exp(-4*fissionRate*timeWindow)*(np.exp(fissionRate*timeWindow)-1)**3
        preNormColinFactorQuadruple = np.exp(-5*fissionRate*timeWindow)*(np.exp(fissionRate*timeWindow)-1)**4
        #print(fissionRate)
        
        colinFactorSum=preNormColinFactorSingle+preNormColinFactorDouble+preNormColinFactorTriple#+preNormColinFactorQuadruple
        oldColinFactorSum=colinFactorSingleOld+colinFactorDoubleOld+colinFactorTripleOld
        #Uused to divide the next 3 lines by colinFactorSum but trying something right now

        colinFactorSingle=preNormColinFactorSingle/colinFactorSum
        colinFactorDouble=preNormColinFactorDouble/colinFactorSum
        colinFactorTriple=preNormColinFactorTriple/colinFactorSum

        #colinFactorSingle,colinFactorDouble,colinFactorTriple=MultipleFactorComputer(fissionRate, RunLength[n], timeWindow)
        #print(fissionRate,timeWindow,"colin factor parameters")
        #print(colinFactorSingle,colinFactorDouble,colinFactorTriple,"colin factors")
        colinFactorQuadruple=preNormColinFactorQuadruple/colinFactorSum
        #print(colinFactorSingle,colinFactorDouble,colinFactorTriple,colinFactorQuadruple)
        #colinFactorQuad = np.exp(-5*fissionRate*timeWindow)*(np.exp(fissionRate*timeWindow)-1)**4
        #print(fissionRate)
        #print(colinFactorSingle,colinFactorSingleOld/oldColinFactorSum,"compare single")
        #print(colinFactorDouble,colinFactorDoubleOld/oldColinFactorSum,"Compare double")
        #print(colinFactorTriple,colinFactorTripleOld/oldColinFactorSum,"compare triple")
        #print(colinFactorQuad)
        
        #colinFactorDouble = np.exp(-3*fissionRate*timeWindow)*(np.exp(fissionRate*timeWindow)-1)**2
        
        #Figure out how many of the fissions are from single/double/triple counts
        prenormsingleCountProportion=SingleProportion(preEffDistTotal,colinFactorSingle)
        #print(prenormsingleCountProportion,"prenorm")
        prenormdoubleCountProportion,doubleForTriple=DoubleProportion(preEffDistTotal,colinFactorDouble,runLengthFactor)
        prenormtripleCountProportion=TripleProportion(preEffDistTotal,colinFactorTriple,doubleForTriple,runLengthFactor)
        psdtsum=[sum(i) for i in zip(prenormsingleCountProportion,prenormdoubleCountProportion,prenormtripleCountProportion)]
        sdtsum=sum(psdtsum)
       # print(sdtsum,"sdtsum")
        singleCountProportion=[x/sdtsum for x in prenormsingleCountProportion]
        doubleCountProportion=[x/sdtsum for x in prenormdoubleCountProportion]
        tripleCountProportion=[x/sdtsum for x in prenormtripleCountProportion]
       # print(sum(singleCountProportion),"singles")
        #print(sum(doubleCountProportion),"dubs")
       # print(sum(tripleCountProportion),"trips")
        #Adds the above, determining for a 100% efficiency detector the distribution of fissions coming from the source
        preEfficiencyDistribution=DistributionAdder(singleCountProportion,doubleCountProportion,tripleCountProportion)
        #These 3 lines are mostly for the graphs, but find how many fissions come from each type of detection
        finalSingle=ComputeEfficiency(singleCountProportion,Eff)
        finalDouble=ComputeEfficiency(doubleCountProportion,Eff)
        finalTriple=ComputeEfficiency(tripleCountProportion,Eff)
        #This finds the total distribution of fissions (before the background is added)

        
        fullDistPreBackground=ComputeEfficiency(preEfficiencyDistribution,Eff)
        #This multiplies the number of fissions detected by the distribution, giving us the absolute distribution
        neutronsDetected=FissionsDetected(totalCurrentFissions,fullDistPreBackground)
        #This adds in the background
        finalWithBackground=BackgroundHandler(neutronsDetected,backGroundEvents,RunLength[n])
        randomFluctuations=1
        if randomFluctuations==1 and PrintResults==True:
            finalDistribution=PoissonFluctuations(finalWithBackground)
        else:
            #finalDistribution=np.round(finalWithBackground)
            finalDistribution=finalWithBackground
            #testdist=PoissonFluctuations(finalWithBackground)
            #print(testdist,"with rando")
            #print((finalDistribution),"without rando")
           # print((f'{finalDistribution:f}'))
            #print(([int(a) for a in finalDistribution]),"without rando")
        totalNeutrons=0
        for i in range(len(finalDistribution)):
            totalNeutrons+=finalDistribution[i]*(i+1)
    
    

        finalDistributionStrange=np.round([totalCurrentFissions*x for x in DistributionAdder(finalSingle,finalDouble,finalTriple)])
        if PrintResults==True and GenerateTotalCounts==False:
            plt.plot(np.round([totalCurrentFissions*x for x in DistributionAdder(finalSingle,finalDouble,finalTriple)]),label="Method that I used before")
            plt.plot(np.round(finalWithBackground),label="Method that should work")
            plt.yscale("log")
            plt.title("Sample aged "+str(Years[n])+" years, Cf252 "+str(Cf252Prop*100)+"% Cf250 "+str(Cf250Prop*100)+"% E= "+str(Eff))
            plt.xticks(np.arange(len(fullDistPreBackground)),np.arange(1,len(fullDistPreBackground)+1))
            plt.xlabel("Multiplicity")        
            plt.ylabel("Fissions")
            plt.legend()
            plt.show()
            np.savetxt("TestDataFissions"+str(Years[n])+".csv",finalDistribution,delimiter=',')

            
            #print(finalDistribution)
            #print(np.round([totalCurrentFissions*x for x in DistributionAdder(finalSingle,finalDouble,finalTriple)]))
            print(finalDistributionStrange,"Without background")
            print(finalDistribution,"With Background")
            #print(sum(finalDistribution),"sum")

            print(totalNeutrons,"Total Neutrons Detected")
            print(totalNeutrons/RunLength[n],"Neutron Rate")
            print("====================================================================")
        #finalDistribution=np.ndarray.tolist(finalDistribution)
        pushToDateFitter.append(finalDistribution)
        if GenerateTotalCounts:
            listOfNeutrons.append(((totalNeutrons)/RunLength[n]))
    if GenerateTotalCounts:
        return(listOfNeutrons)       
    else:
        return(pushToDateFitter)
    
    
def FissionProportions(massProportion1,massProportion2,decayConstant1,decayConstant2,channelProp1,channelProp2):
    _fissionProportion252=(massProportion1*decayConstant1*channelProp1)/(massProportion1*decayConstant1*channelProp1+massProportion2*decayConstant2*channelProp2)
    _fissionProportion250=(massProportion2*decayConstant2*channelProp2)/(massProportion1*decayConstant1*channelProp1+massProportion2*decayConstant2*channelProp2)
    #_fissionProportion248=(massProportion3*decayConstant3*channelProp3)/(massProportion1*decayConstant1*channelProp1+massProportion2*decayConstant2*channelProp2+massProportion3*decayConstant3*channelProp3)
    return _fissionProportion252,_fissionProportion250#,_fissionProportion248

def InitialFissionComputer(Prop,Fissions):
    return(Prop*Fissions)

def FissionComputer(initialFissions,DecayConst,Year,runLengthFactor):
    return(runLengthFactor*initialFissions*np.exp(-DecayConst*Year))

def FissionComputer248(Initial252,Current252,DecayConst252,DecayConst248,Chan252Fission,Chan252Alpha,Chan248Fission):#,FissFrom248Initial):
    _initialmass252=Initial252/(DecayConst252*Chan252Fission)
    _currentmass252=Current252/(DecayConst252*Chan252Fission)
    _lostmass252=_initialmass252-_currentmass252
    _currentmass248=_lostmass252*Chan252Alpha
    _currentfiss248=(_currentmass248*DecayConst248*Chan248Fission)#+FissFrom248Initial
    return(_currentfiss248)

def FissionRate(Fissions,Time):
    _fissionrate=Fissions/Time
    return _fissionrate
def DistComputer(Fiss252,Fiss250,Fiss248,FissMystery,Dist252,Dist250,Dist248):
    FissProp252=Fiss252/(Fiss252+Fiss250+Fiss248+FissMystery)
    FissProp250=Fiss250/(Fiss252+Fiss250+Fiss248+FissMystery)
    FissProp248=Fiss248/(Fiss252+Fiss250+Fiss248+FissMystery)
    FissPropMystery=FissMystery/(Fiss252+Fiss250+Fiss248+FissMystery)
    
    _dist252=[FissProp252*x for x in Dist252]
    _dist250=[FissProp250*x for x in Dist250]
    _dist248=[FissProp248*x for x in Dist248]
    _distMystery=[FissPropMystery*x for x in Dist250]
    
    _disttotal=[]
    for (item1,item2,item3,item4) in zip(_dist252,_dist250,_dist248,_distMystery):
        _disttotal.append(item1+item2+item3+item4)
    _disttotal=[x/sum(_disttotal) for x in _disttotal]    
    return _disttotal

def SingleProportion(Dist,Factor):
    _newProportion=[]
    _newProportion=[Factor*x for x in Dist]
    return _newProportion

def DoubleProportion(Dist,Factor,RunLengthFactor):
    _doubleCount=[0]*len(Dist)
    for i in range(len(Dist)-1):
        for j in range(i+1):
            _doubleCount[i+1]+=Dist[j]*Dist[i-j]
    _newProportion=[]
    _newProportion=[Factor*x for x in _doubleCount]
    _newProportionTest=[]
    _newProportionTest=[RunLengthFactor*x for x in _newProportion]
    return _newProportion,_newProportionTest
def TripleProportion(Dist,Factor,DoubleCountDistribution,RunLengthFactor):
    _tripleCount=[0]*len(DoubleCountDistribution)
    for i in range(len(DoubleCountDistribution)-1):
        for j in range(i+1):
            _tripleCount[i+1]+=DoubleCountDistribution[j]*Dist[i-j]
    _newProportion=[]
    _newProportion=[Factor*x for x in _tripleCount]
    _newProportionTest=[]
    _newProportionTest=[RunLengthFactor*x for x in _newProportion]
    return _newProportionTest
    
def DistributionAdder(Single,Double,Triple):
    _completeDistribution=[]
    for (item1,item2,item3) in zip(Single,Double,Triple):
        _completeDistribution.append(item1+item2+item3)
    return _completeDistribution

def ComputeEfficiency(Distribution,Efficiency):
    outputArray=np.zeros(len(Distribution)+1)
    for i in range(len(Distribution)):
        j=i
        while j<len(Distribution):
            outputArray[i]+=comb(j+1,i+1)*Efficiency**(i+1)*((1-Efficiency)**(j+1-(i+1)))*Distribution[j]
            
            j=j+1
    return outputArray

def FissionsDetected(Fissions,Dist):
    _a=([x*Fissions for x in Dist])
    return _a

def BackgroundHandler(Dist,Background,RunLength):
    _scalingFactor=(Background)/sum(Dist)
    _BackGround=[]
    _BackGround.append(Background)
    if(_scalingFactor>=1):
        print("The number of backgorund events exceeds the number of events from the source. Try increasing the mass number passed into the data generation function if using generated data.")
        #raise Exception("The number of background events exceed the number of events from the source. Try increasing the mass number passed in.")
    _scaled=[(1-_scalingFactor)*x for x in Dist]
    _scaled2=[x+y for x,y in zip_longest(_scaled,_BackGround,fillvalue=0)]
    return _scaled2

def PoissonFluctuations(Distribution):
    _fluctuatedDistribution=[]
    for i in range(len(Distribution)):
        _fluctuatedDistribution.append(np.random.poisson(Distribution[i]))
    return _fluctuatedDistribution

def ConvertNeutronsToMassProportions(values252,values250,decayConst252,decayConst250,channelProportion252Fission,channelProportion250Fission,yearOfInterest):
    fissprop252=(values252/(values252+values250))
    fissprop250=(values250/(values252+values250))
    
    protomassprop252=fissprop252/(decayConst252*channelProportion252Fission)
    protomassprop250=fissprop250/(decayConst250*channelProportion250Fission)
    
    massprop252=protomassprop252/(protomassprop252+protomassprop250)
    massprop250=protomassprop250/(protomassprop252+protomassprop250)
    
    return massprop252,massprop250


def Cf250MassFractionGrabber(NCf252,NCf250,decayConst252,decayConst250,chanFiss252,chanFiss250,year):
# =============================================================================
#     fissprop252=NCf252/(NCf250+NCf252)
#     fissprop250=NCf250/(NCf250+NCf252)
#     
#     m252=fissprop252/(decayConst252*chanFiss252)
#     m250=fissprop250/(decayConst250*chanFiss250)
#     
#     m252_norm=m252/(m252+m250)
#     m250_norm=m250/(m252+m250)
#     
#     frac=m250_norm/m252_norm
# =============================================================================

    fiss252=NCf252*np.exp(decayConst252*year)
    fiss250=NCf250*np.exp(decayConst250*year)
    
    m252=fiss252/(decayConst252*chanFiss252)
    m250=fiss250/(decayConst250*chanFiss250)
    frac=m250/m252

    #print(m252,"m252")
    #print(m250,"m250")
    #print(frac,"frac")
    return frac
    
    
    
#DataGenerator([21.5425,21.6705,21.7885,21.8655,27.0242],75207614,0.468606688,0.889515,0.110485,[14402,14402,14402,14402,25786,25786,25786],[0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671],True,False)
#DataGenerator([20.5589,20.6869,20.8049,20.8819,26.0406,26.3557],30185601,0.482443754,0.262445628,1.-0.262445628,[14402,14402,14402,14402,25786,25786],[0.00671,0.0671,0.0671,0.0671,0.0671,0.0671],True,False)
#qq=DataGenerator([16.0630],42075299,0.482499012,.317574077,1-.317574077,[14402,14402,14402,14402,25786,25786,25786],[0.00671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671],True,False,JustFissions=False)
#qq2=DataGenerator([21.8438],38974823,0.483709823,.301655128,1-.301655128,[14402],[0.00671],True,False)
#qq2=DataGenerator([21.4603, 21.8438, 21.9726, 22.0932, 22.1699, 27.326],38974823,0.483709823,0.5,0.5,1000.,[4801,14402,14402,14402,14402,25786],[0.075,0.0671,0.0671,0.0671,0.0671,0.11],True,False)
#qq2=DataGenerator([21.4603, 21.8438, 21.9726, 22.0932, 22.1699, 27.326,27.8],176609750,0.4656574,0.8,0.2,1001,0.0261, 0.127, 0.274, 0.3045, 0.1852, 0.0658, 0.0154, 0.002,[4801,14402,14402,14402,14402,25786,25786],[0.02,0.02,0.02,0.02,0.02,0.02,0.02],True,False)
def BandCreator(values,resolution,start,end):
    rangeOfValues=np.linspace(start,end,resolution)
    tripleExp=True
    #neutrons=DataGenerator(rangeOfValues,values[0],values[1],1/(1+values[2]),values[2],values[3],resolution*[14402],resolution*[0.02],True,True,JustFissions=False)
    neutrons=(DataGenerator(rangeOfValues,values[0],values[1],1/(1+values[2]),values[2],values[3],resolution*[4801,14402,14402,14402,14402,25786,25786],resolution*[0.02],True,True,JustFissions=False))
    if tripleExp:
        
        def NegativeExp(x,a,b,c,d,f,g,h):
            return a*np.exp(-b*x)+c*np.exp(-d*x)+f*np.exp(-g*x)+h
        p0=[500,1,50,1,30,1,1]
    
        popt,pcov=curve_fit(NegativeExp,rangeOfValues,neutrons,p0,maxfev=800000)
        print(pcov)
        list_dict = {'Years':rangeOfValues, 'Neutron Rate':neutrons}
        df = pd.DataFrame(list_dict) 
        df.to_csv('listOfValues.csv', index=False) 
        print(popt,"values")
        analyticInterp=[]
        for i in range(len(rangeOfValues)):
            analyticInterp.append(popt[0]*np.exp(-popt[1]*rangeOfValues[i])+popt[2]*np.exp(-popt[3]*rangeOfValues[i])+popt[4]*np.exp(-popt[5]*rangeOfValues[i])+popt[6])
        
        plt.plot(rangeOfValues,analyticInterp,color='r')
        plt.xlabel("Time (Years)")
        plt.ylabel("Fissions/Second")
        plt.title("Fission Rate for the next few Years")
        plt.scatter(rangeOfValues,neutrons)
        #plt.scatter([20.5589,20.6869,20.8049,20.8819,26.0406],[16.415289,16.101305,15.844049,15.654701,8.288102],color='gold')
        #Numbers in second list are neutrons per second deetected
        plt.scatter([21.4603, 21.8438, 21.9726, 22.0932, 22.1699, 27.326,27.8],[i/1 for i in [36.15,34.01,33.25,32.72,32.37,16.47,16.04]],color='gold')
        plt.show()

if __name__=="__main__":
    BandCreator([17485934,0.477,0.0536,23.89],100,20,28)
    #qq2=DataGenerator([21.4603],17485934,0.477155,1/(1+0.0536),1-(1/(1+0.0536)),23.88989797,[14402],[0.02],True,False)
    #qq2=DataGenerator([21.4603, 21.8438, 21.9726, 22.0932, 22.1699, 27.326,27.8],17485934,0.47715464583109934,0.94913,1-0.94913,23.89,[4801,14402,14402,14402,14402,25786,25786],[0.02,0.02,0.02,0.02,0.02,0.02,0.02],True,False)
    #print(qq2,"qq2")

#print(sum(qq[0])/14402,"qq")

#print((sum(qq[0])/14402)/qq2[0])
#print(sum(qq),"sum qq")
doGenFunc=False




if doGenFunc:      
    resolution=500
    rangeOfValues=np.linspace(20,50,resolution)   
    #DataGenerator([33.932,34.06,34.178,34.255,39.4137,39.6,40],932560648,0.486700153,0.856340985,0.143659015,[14402,14402,14402,14402,25786,25786,25786],[0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671],True,True)
    #Change rate - it is detected not emitted
    tripleExp=True
    neutronRange=(DataGenerator(rangeOfValues,17622772,0.4762,0.9490,1-0.9490,23.576,resolution*[4801,14402,14402,14402,14402,25786,25786],resolution*[0.02],True,True,JustFissions=False))
    #neutronRange=[x for x in protoneutronRange]
    print(neutronRange[0],"min max")
    if tripleExp:
        
        def NegativeExp(x,a,b,c,d,f,g,h):
            return a*np.exp(-b*x)+c*np.exp(-d*x)+f*np.exp(-g*x)+h
        p0=[500,1,50,1,30,1,1]
    
        popt,pcov=curve_fit(NegativeExp,rangeOfValues,neutronRange,p0,maxfev=50000)
        print(pcov)
        list_dict = {'Years':rangeOfValues, 'Neutron Rate':neutronRange}
        df = pd.DataFrame(list_dict) 
        df.to_csv('listOfValues.csv', index=False) 
        print(popt,"values")
        analyticInterp=[]
        for i in range(len(rangeOfValues)):
            analyticInterp.append(popt[0]*np.exp(-popt[1]*rangeOfValues[i])+popt[2]*np.exp(-popt[3]*rangeOfValues[i])+popt[4]*np.exp(-popt[5]*rangeOfValues[i])+popt[6])
        
        plt.plot(rangeOfValues,analyticInterp,color='r')
        plt.xlabel("Time (Years)")
        plt.ylabel("Fissions/Second")
        plt.title("Fission Rate for the next few Years")
        plt.scatter(rangeOfValues,neutronRange)
        #plt.scatter([20.5589,20.6869,20.8049,20.8819,26.0406],[16.415289,16.101305,15.844049,15.654701,8.288102],color='gold')
        plt.scatter([21.4603, 21.8438, 21.9726, 22.0932, 22.1699, 27.326,27.8],[36.15,34.01,33.25,32.72,32.37,16.47,16.04],color='gold')
        plt.show()
    
    if tripleExp==False:
        def NegativeExp(x,a,b,c,d,f):
            return a*np.exp(-b*x)+c*np.exp(-d*x)+f
        p0=[1600,np.log(2)/2.65,400,np.log(2)/2.65,1]
    
        popt,pcov=curve_fit(NegativeExp,rangeOfValues,neutronRange,p0,maxfev=50000)
        print(pcov)
        list_dict = {'Years':rangeOfValues, 'Neutron Rate':neutronRange}
        df = pd.DataFrame(list_dict) 
        df.to_csv('listOfValues.csv', index=False) 
        print(popt,"values")
        analyticInterp=[]
        for i in range(len(rangeOfValues)):
            analyticInterp.append(popt[0]*np.exp(-popt[1]*rangeOfValues[i])+popt[2]*np.exp(-popt[3]*rangeOfValues[i])+popt[4])
        
        plt.plot(rangeOfValues,analyticInterp,color='r')
        plt.xlabel("Time (Years)")
        plt.ylabel("Fissions/Second")
        plt.title("Fission Rate for the next 10 Years")
        plt.scatter(rangeOfValues,neutronRange)
        #plt.scatter([20.5589,20.6869,20.8049,20.8819,26.0406],[16.415289,16.101305,15.844049,15.654701,8.288102],color='gold')
        plt.scatter([21.8438,21.9718,22.0898,22.1668,27.3255],[16.415289,16.101305,15.844049,15.654701,8.288102],color='gold')
    
        plt.show()
# =============================================================================
# if doGenFunc:      
#     resolution=500
#     rangeOfValues=np.linspace(0,40,resolution)   
#     #DataGenerator([33.932,34.06,34.178,34.255,39.4137,39.6,40],932560648,0.486700153,0.856340985,0.143659015,[14402,14402,14402,14402,25786,25786,25786],[0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671],True,True)
#     #Change rate - it is detected not emitted
#     tripleExp=False
#     neutronRange=(DataGenerator(rangeOfValues,42075299,0.482499012,0.317574077,1-.317574077,resolution*[14402],resolution*[0.0671],True,True,JustFissions=True))
#     if tripleExp:
#         
#         def NegativeExp(x,a,b,c,d,f,g,h):
#             return a*np.exp(-b*x)+c*np.exp(-d*x)+f*np.exp(-g*x)+h
#         p0=[500,1,50,1,30,1,1]
#     
#         popt,pcov=curve_fit(NegativeExp,rangeOfValues,neutronRange,p0,maxfev=50000)
#         print(pcov)
#         list_dict = {'Years':rangeOfValues, 'Neutron Rate':neutronRange}
#         df = pd.DataFrame(list_dict) 
#         df.to_csv('listOfValues.csv', index=False) 
#         print(popt,"values")
#         analyticInterp=[]
#         for i in range(len(rangeOfValues)):
#             analyticInterp.append(popt[0]*np.exp(-popt[1]*rangeOfValues[i])+popt[2]*np.exp(-popt[3]*rangeOfValues[i])+popt[4]*np.exp(-popt[5]*rangeOfValues[i])+popt[6])
#         
#         plt.plot(rangeOfValues,analyticInterp,color='r')
#         plt.xlabel("Time (Years)")
#         plt.ylabel("Fissions/Second")
#         plt.title("Fission Rate for the next few Years")
#         plt.scatter(rangeOfValues,neutronRange)
#         #plt.scatter([20.5589,20.6869,20.8049,20.8819,26.0406],[16.415289,16.101305,15.844049,15.654701,8.288102],color='gold')
#         plt.scatter([21.8438,21.9718,22.0898,22.1668,27.3255],[16.415289,16.101305,15.844049,15.654701,8.288102],color='gold')
#     
#         plt.show()
#     
#     if tripleExp==False:
#         def NegativeExp(x,a,b,c,d,f):
#             return a*np.exp(-b*x)+c*np.exp(-d*x)+f
#         p0=[1600,np.log(2)/2.65,400,np.log(2)/2.65,1]
#     
#         popt,pcov=curve_fit(NegativeExp,rangeOfValues,neutronRange,p0,maxfev=50000)
#         print(pcov)
#         list_dict = {'Years':rangeOfValues, 'Neutron Rate':neutronRange}
#         df = pd.DataFrame(list_dict) 
#         df.to_csv('listOfValues.csv', index=False) 
#         print(popt,"values")
#         analyticInterp=[]
#         for i in range(len(rangeOfValues)):
#             analyticInterp.append(popt[0]*np.exp(-popt[1]*rangeOfValues[i])+popt[2]*np.exp(-popt[3]*rangeOfValues[i])+popt[4])
#         
#         plt.plot(rangeOfValues,analyticInterp,color='r')
#         plt.xlabel("Time (Years)")
#         plt.ylabel("Fissions/Second")
#         plt.title("Fission Rate for the next 10 Years")
#         plt.scatter(rangeOfValues,neutronRange)
#         #plt.scatter([20.5589,20.6869,20.8049,20.8819,26.0406],[16.415289,16.101305,15.844049,15.654701,8.288102],color='gold')
#         plt.scatter([21.8438,21.9718,22.0898,22.1668,27.3255],[16.415289,16.101305,15.844049,15.654701,8.288102],color='gold')
#     
#         plt.show()
# =============================================================================
    #DataGenerator([34],1500000000,0.48,0.15,[14402],[0.0671],False)
    #DataGenerator([34.918008706717764],1334224159,0.4799469847017151,0.8483755881470229,1-0.8483755881470229,[14402],[0.0671],True)
    #DataGenerator([20,21,25,27,29,30,31,32,33,34,35,36,37,42,45],1650000000,0.48,0.85,0.15,[100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000],[0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671],True)
    #QUESTION FOR TOM: It seems like the graph that DOESN'T include the background radiation fits to the real data SUBSTANTIALLY better
    
    #May 17th, 1996 is when it was sent off.
    #May 24th, 1995 is when it was created. (21.5425 years before the initial run)

#21.5425