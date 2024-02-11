import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from itertools import zip_longest
import Double_Count_Distributor as MultiCounter
import scipy.optimize
import numdifftools
import time

start_time=time.time()
def DataGenerator(Years,protoFissions,Eff,Cf252Prop,Cf250Prop,RunLength,doTheMLE):
    #Defining Constants
    yearOfInterest=Years
    Efficiency=Eff
    massProportion252=Cf252Prop
    massProportion250=Cf250Prop
    decayConst252=np.log(2)/2.65
    decayConst250=np.log(2)/13.08
    channelProportion252Alpha=0.96908
    channelProportion252Fission=1-channelProportion252Alpha
    channelProportion250Alpha=0.99923
    channelProportion250Fission=1-channelProportion250Alpha
    
    timeWindow=0.0015#Time window in seconds,  might be increased to 0.002 in the future
    multi252=3.735
    multi250=3.50
    #This is for run 5557
    #protoBackgroundEvents=[5806,30,10,1]
    protoBackgroundEvents=[14400./60.,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    backgroundRunLength=14400
        #colinFactorSingle=np.exp(-2*Fissions*timeWindow/runLength)
        #colinFactorDouble=3*np.exp(-Fissions*timeWindow/runLength)-2*np.exp(-2*Fissions*timeWindow/runLength)-1
        #colinFactorTriple=(1-np.exp(-Fissions*timeWindow/runLength))**2
    #cf252MultiNoNorm=[0.02688,0.12366,0.26936,0.29947,0.18172,0.06346,0.0113,0.0032258,0.00115,0,0,0,0,0]
    #cf250NultiNoNorm=[0.03594,0.16771,0.29635,0.30104,0.14531,0.04740,0.00417,0.00208,0.001,0,0,0,0,0]
    #cf252MultiNoNorm=[0.02688,0.12366,0.26936,0.29947,0.18172,0.06346,0.0113,0.0032258,0,0,0,0,0,0]
    #before colin's probs ^^
    cf252MultiNoNorm=[0.026,0.1267,0.2734,0.3039,0.1848,0.0657,0.0154,0.0020,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #Unit Testing Parameters
    #cf252MultiNoNorm=[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #cf250MultiNoNorm=[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    cf250MultiNoNorm=[0.03594,0.16771,0.29635,0.30104,0.14531,0.04740,0.00417,0.00208,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #cf252MultiNoNorm=[0,0,1,0,0,0,0,0,0,0,0,0,0,0]
    Cf252Multi=[x / sum(cf252MultiNoNorm) for x in cf252MultiNoNorm]
    Cf250Multi=[x / sum(cf250MultiNoNorm) for x in cf250MultiNoNorm]
    arrayToPushToMLE=[]
    _atest=[]
    _ctest=[]
    for n in range(len(yearOfInterest)):
        if len(RunLength)==1:
            #Assume the "default" run would be 4 hours
            protorunLengthFactor=14400
        else:
            protorunLengthFactor=RunLength[0]
            
        runLengthFactor=RunLength[n]/protorunLengthFactor
        #runLengthFactor=1
        Fissions=protoFissions
        runLength=RunLength
        backGroundToData=runLength[n]/backgroundRunLength
        backgroundEvents=[(x*backGroundToData) for x in protoBackgroundEvents]
        backgroundRate=[x / backgroundRunLength for x in backgroundEvents]

        #print(Cf252Prop,Cf250Prop)
        initialFissionProportion252,initialFissionProportion250=FissionProportions(massProportion252,massProportion250,decayConst252,decayConst250,channelProportion252Fission,channelProportion250Fission)

        initialFissions252=InitialFissionComputer(Fissions,initialFissionProportion252,decayConst252,yearOfInterest[n])
        initialFissions250=InitialFissionComputer(Fissions,initialFissionProportion250,decayConst250,yearOfInterest[n])
        currentFissions252=CurrentFissionComputer(initialFissions252,decayConst252,yearOfInterest[n],runLengthFactor)
        currentFissions250=CurrentFissionComputer(initialFissions250,decayConst250,yearOfInterest[n],runLengthFactor)
        print(currentFissions252,currentFissions250,"currentfissions")
        
        totalCurrentFissions=(currentFissions252)+(currentFissions250)
        fissionRate=FissionRate(totalCurrentFissions,runLength[n])
        earlyDist252=EarlyDistComputer(currentFissions252,Cf252Multi)
        earlyDist250=EarlyDistComputer(currentFissions250,Cf250Multi)
        preEffDistTotal=DistComputer(currentFissions252,currentFissions250,Cf252Multi,Cf250Multi)
        
        #singleCountProportionNaive=preEffDistTotal
        #doubleCountProportionNaive=DoubleCountDistribution(preEffDistTotal)
        #tripleCountProportionNaive=TripleCountDistribution(preEffDistTotal,doubleCountProportionNaive)
        absoluteDoubleCountsForGraph=DoubleCountComputer(fissionRate,timeWindow,runLength[n])
        absoluteTripleCountsForGraph=TripleCountComputer(fissionRate,timeWindow,runLength[n])
        
        #singleCountProportion=ProportionWisener(colinFactorSingle,singleCountProportionNaive)
        #doubleCountProportion2=ProportionWisener(colinFactorDouble,doubleCountProportionNaive)
        #tripleCountProportion=ProportionWisener(colinFactorTriple,tripleCountProportionNaive)

        colinFactorSingle=np.exp(-2*totalCurrentFissions*timeWindow/runLength[n])
        colinFactorDouble=3*np.exp(-totalCurrentFissions*timeWindow/runLength[n])-2*np.exp(-2*totalCurrentFissions*timeWindow/runLength[n])-1
        colinFactorTriple=(1-np.exp(-totalCurrentFissions*timeWindow/runLength[n]))**2
        #Should triple and double count proportions take in singlecountproportions instead of preeffdisttotal?
        singleCountProportion=SingleProportion(preEffDistTotal,colinFactorSingle)
        doubleCountProportion,doubleForTriple=DoubleProportion(preEffDistTotal,colinFactorDouble,runLengthFactor)
        tripleCountProportion=TripleProportion(preEffDistTotal,colinFactorTriple,doubleForTriple,runLengthFactor)
        #print(fissionRate,"fissionrate")
        #print(sum(singleCountProportion),sum(doubleCountProportion),sum(tripleCountProportion),"stats")
        #print(Cf252Prop,Cf250Prop)
        #fullDistribution=DistributionAdder(singleCountProportion252,singleCountProportion250,doubleCountProportion252,doubleCountProportion250,tripleCountProportion252,tripleCountProportion250)
        fullDistributionBeforeEfficiency=DistributionAdder(singleCountProportion,doubleCountProportion,tripleCountProportion)
        #fullDistributionBeforeEfficiency=[x/sum(protofullDistributionBeforeEfficiency) for x in protofullDistributionBeforeEfficiency]
        
        singdistforgraph,dubdistforgraph,tripdistforgraph=((ComputeEfficiency2(singleCountProportion,Efficiency))),(ComputeEfficiency2(doubleCountProportion,Efficiency)),(ComputeEfficiency2(tripleCountProportion,Efficiency))
        fullDistributionPreBackground=ComputeEfficiency2(fullDistributionBeforeEfficiency,Efficiency)
        
        #fullDistribution=fullDistributionBeforeEfficiency
        #normalizedDistribution=[x/sum(fullDistributionPreBackground) for x in fullDistributionPreBackground]
        #print(normalizedDistribution,"unit test")
        #a=([x*totalCurrentFissions for x in normalizedDistribution])
        a=([x*totalCurrentFissions for x in fullDistributionPreBackground])
        b=BackgroundHandler(a,backgroundEvents,runLength[n])
        if doTheMLE==False:
            c=PoissonFluctuations(b)
            #c=b
            #c=np.round(b)
        else:
            c=np.round(b)
            #if any(x<0 for x in b)==True:
                #c=np.round(b)
            #else:
                #c=PoissonFluctuations(b)
                #c=np.round(b)
                
             #   c=np.round(b)
                #c=PoissonFluctuations(b)
        totalNeutrons=0
        for i in range(len(c)):
            totalNeutrons+=(i+1)*c[i]
        #fig=plt.figure()
        #ax=fig.add_subplot(1,1,1)
        
        
        #plt.plot(np.round(c),label="Total Distribution")
        #plt.plot(np.round([totalCurrentFissions*x/(1-sum(backgroundEvents)/totalCurrentFissions) for x in tripdistforgraph]),label="Triple Fission Counts")
        #plt.plot(np.round([totalCurrentFissions*x/(1-sum(backgroundEvents)/totalCurrentFissions) for x in dubdistforgraph]),label="Double Fission Counts")
        #plt.plot(np.round([totalCurrentFissions*x/(1-sum(backgroundEvents)/totalCurrentFissions) for x in singdistforgraph]),label="Single Fission Counts")
        #plt.plot(np.round(backgroundEvents),label="Background Event Counts")
       
        
# =============================================================================
        if doTheMLE==False:
            plt.plot(np.round([totalCurrentFissions*x for x in DistributionAdder(tripdistforgraph, dubdistforgraph, singdistforgraph)]),label="summed method 2")
            #print(np.round([totalCurrentFissions*x for x in DistributionAdder(tripdistforgraph, dubdistforgraph, singdistforgraph)]),"new way")
            plt.yscale("log")
            #plt.plot(_finalDistribution)
            plt.title("Sample aged "+str(yearOfInterest[n])+" years, Cf252 "+str(Cf252Prop*100)+"% Cf250 "+str(Cf250Prop*100)+"% E= "+str(Eff))
            plt.xticks(np.arange(len(fullDistributionPreBackground)),np.arange(1,len(fullDistributionPreBackground)+1))
            plt.xlabel("Multiplicity")
            #plt.xlim(-1,15)
            plt.ylabel("Fissions")
            print(yearOfInterest[n])
            print(np.round(c),"Full Data")
            print(np.round(b),"Full Data No Random")
            print(sum(b),"No random sum")
            print(sum(c),"Sum")
            plt.legend()
            plt.show()
            np.savetxt("TestDataFissions"+str(yearOfInterest[n])+".csv",c,delimiter=',')
#         
#         
        _ctest.append(c)
        
        #_ctest.append(c)
       # print(_atest,"bntest")
        #print(_ctest,"ctest")
        #print(_atest,"ATEST IN FUNCTION")
        #print(_atest,"a test in function")
# =============================================================================
        
    #print(_ctest,"ctest")
    return _ctest

    
    
    

def FissionProportions(massProportion1,massProportion2,decayConstant1,decayConstant2,channelProp1,channelProp2):
    _fissionProportion252=(massProportion1*decayConstant1*channelProp1)/(massProportion1*decayConstant1*channelProp1+massProportion2*decayConstant2*channelProp2)
    _fissionProportion250=(massProportion2*decayConstant2*channelProp2)/(massProportion1*decayConstant1*channelProp1+massProportion2*decayConstant2*channelProp2)
    return _fissionProportion252,_fissionProportion250 

def FissionRate(Fissions,RunLength):
    _fissionRate=Fissions/RunLength
    return(_fissionRate)    

def EarlyDistComputer(Value,Dist):
    _output=[Value*x for x in Dist]
    return _output

def InitialFissionComputer(Fissions,InitialProportion,DecayConst,Year):
    _initialFissions=Fissions*InitialProportion
    #_currentFissions=_initialFissions*np.exp(-DecayConst*Year)
    return(_initialFissions)
def CurrentFissionComputer(InitialFissions,DecayConst,Year,RunLengthFactor):
    _currentFissions=RunLengthFactor*InitialFissions*np.exp(-DecayConst*Year)
    return _currentFissions

def DistComputer(CurrentFissions252,CurrentFissions250,Cf252Multi,Cf250Multi):
    fissionProportion252=(CurrentFissions252/(CurrentFissions250+CurrentFissions252))
    fissionProportion250=1-fissionProportion252
        
    preEffDistTotal252=[fissionProportion252*x for x in Cf252Multi]
    preEffDistTotal250=[fissionProportion250*x for x in Cf250Multi]

    preEffDistTotal=[]
    for (item1,item2) in zip(preEffDistTotal252,preEffDistTotal250):
        preEffDistTotal.append(item1+item2)

    preEffDistTotal=[x/sum(preEffDistTotal) for x in preEffDistTotal]

    return preEffDistTotal

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

def ProportionWisener(Factor,Proportions):
    _newProportion=[]
    _newProportion=[Factor*x for x in Proportions]
    return _newProportion

def TripleProportion(Dist,Factor,DoubleCountDistribution,RunLengthFactor):
    _tripleCount=[0]*len(DoubleCountDistribution)
    for i in range(len(DoubleCountDistribution)-1):
        for j in range(i+1):
            #Ask tom about the factor of 2 here. It is a new addition that I added as a result of the unit test and may not have a place here
            _tripleCount[i+1]+=DoubleCountDistribution[j]*Dist[i-j]
    _newProportion=[]
    _newProportion=[Factor*x for x in _tripleCount]
    _newProportionTest=[]
    _newProportionTest=[RunLengthFactor*x for x in _newProportion]
    return _newProportionTest


def DoubleCountDistribution(Cf252Multi):
    _doubleCount=[0]*len(Cf252Multi)
    for i in range(len(Cf252Multi)-1):
        for j in range(i+1):
            _doubleCount[i+1]+=Cf252Multi[j]*Cf252Multi[i-j]
    return(_doubleCount)
def TripleCountDistribution(Cf252Multi,DoubleCountDistribution):
    _tripleCount=[0]*len(DoubleCountDistribution)
    for i in range(len(DoubleCountDistribution)-1):
        for j in range(i+1):
            _tripleCount[i+1]+=DoubleCountDistribution[j]*Cf252Multi[i-j]
    return _tripleCount

#def DistributionAdder(Distribution1,Distribution2):
 #   _completeDistribution=[]
  #  for (item1,item2) in zip(Distribution1,Distribution2):
   #     _completeDistribution.append(item1+item2)
    #return _completeDistribution

def SingleCountComputer(FissionRate,TimeWindow,runLength):
    _SingleCounts=FissionRate*np.exp(-FissionRate*TimeWindow)*runLength
    return _SingleCounts
def DoubleCountComputer(FissionRate,TimeWindow,runLength):
    _DoubleCounts=FissionRate**2*TimeWindow*np.exp(-FissionRate*TimeWindow)*runLength
    return _DoubleCounts
def TripleCountComputer(FissionRate,TimeWindow,runLength):
    _TripleCounts=0.5*FissionRate**3*TimeWindow**2*np.exp(-FissionRate*TimeWindow)*runLength
    return _TripleCounts
def FissionDistributor(Proportion,Fissions):
    _distributedFissions=[Fissions*x for x in Proportion]
    return _distributedFissions
def DistributionAdder(Dist1,Dist2,Dist3):
    _completeDistribution=[]
    for (item1,item2,item3) in zip(Dist1,Dist2,Dist3):
        _completeDistribution.append(item1+item2+item3)
    return _completeDistribution

def ComputeEfficiency(Distribution,Efficiency,Fissions):
    outputArray=np.zeros(len(Distribution)+1)
    for i in range(len(Distribution)):
        j=i
        while j<len(Distribution):
            outputArray[i]+=comb(j+1,i+1)*Efficiency**(i+1)*((1-Efficiency)**(j+1-(i+1)))*Distribution[j]
            j=j+1
    fissionArray=[]
    for k in range(len(outputArray)):
        fissionArray.append(outputArray[k]*Fissions)
    return fissionArray, outputArray
def ComputeEfficiency2(Distribution,Efficiency):
    outputArray=np.zeros(len(Distribution)+1)
    for i in range(len(Distribution)):
        j=i
        while j<len(Distribution):
            outputArray[i]+=comb(j+1,i+1)*Efficiency**(i+1)*((1-Efficiency)**(j+1-(i+1)))*Distribution[j]
            
            j=j+1
    return outputArray

def BackgroundHandler(Distribution,BackgroundEvents,RunLength):
    _scalingFactor=sum(BackgroundEvents)/sum(Distribution)
    _scaled=[(1-_scalingFactor)*x for x in Distribution]
    _scaled2=[x+y for x,y in zip_longest(_scaled,BackgroundEvents,fillvalue=0)]
    return _scaled2

def PoissonFluctuations(Distribution):
    _fluctuationDistribution=[]
    for i in range(len(Distribution)):
        _fluctuationDistribution.append(np.random.poisson(Distribution[i]))
    return _fluctuationDistribution

def ConvertNeutronsToMassProportions(values252,values250,decayConst252,decayConst250,channelProportion252Fission,channelProportion250Fission,yearOfInterest):
    fissprop252=(values252/(values252+values250))
    fissprop250=(values250/(values252+values250))
    protomassprop252=fissprop252/(decayConst252*channelProportion252Fission)
    protomassprop250=fissprop250/(decayConst250*channelProportion250Fission)
    #protoinitmassprop252=protomassprop252*np.exp(decayConst252*0)
    #protoinitmassprop250=protomassprop250*np.exp(decayConst250*0)
    massprop252=protomassprop252/(protomassprop252+protomassprop250)
    massprop250=protomassprop250/(protomassprop252+protomassprop250)
    return massprop252,massprop250

def PropogateErrorMass(values252,values250,error252,error250,decay252,decay250,chanprop252,chanprop250,year):
    
    fissproperror252=np.sqrt((((values250*error252)/((values252+values250)**2))**2)+((values252*error250)/((values252+values250)**2))**2)
    fissproperror250=fissproperror252
    fissprop252=(values252/(values252+values250))
    fissprop250=(values250/(values252+values250))
    print(fissproperror252,"FISPROPERROR (step 1)")
    protomassprop252=fissprop252/(decay252*chanprop252)
    protomassprop250=fissprop250/(decay250*chanprop250)
    protomassproperror252=fissproperror252/(decay252*chanprop252)
    print(protomassproperror252,"PROTOMASSPROPERROR (step 2)")
    protomassproperror250=fissproperror250/(decay250*chanprop250)
    print(protomassproperror252,protomassprop252,"MASS PROP ERROR AND MASS PROP 252")
    print(protomassproperror250,protomassprop250,"MASS PROP ERROR AND MASS PROP 250")
    productforerror1=protomassprop252*protomassproperror250
    productforerror2=protomassprop250*protomassproperror252
    squaresumforerror=(protomassprop252+protomassprop250)**2
    massproperror252=np.sqrt((productforerror1/squaresumforerror)**2+(productforerror2/squaresumforerror)**2)
    massproperror250=np.sqrt((((protomassprop252*protomassproperror250)/((protomassprop250+protomassprop252)**2))**2)+(((protomassprop250*protomassproperror252)/((protomassprop250+protomassprop252)**2))**2))
    print(massproperror252,"massproperror252 (Step 3)")
    
    return massproperror252,massproperror250






#print(DataGenerator(Years,initialAmount,Eff,Cf252,Cf250,[14400,14400,14400,14400,]),"Data Generation Test NEw Years")

#DataGenerator([5],[100000],.9,.7,.3,[2000])


#From The meeting on March 21st, when integrating total fissions & fitting, must look at distribution of each run as well as the total integrated fissions
########DataGenerator([16.932,17.06,17.178,17.255,22.255],22700000,.4715,.85,.15,[14400,14400,14400,14400,259200])

#DataGenerator([12.7,12.7],10000000,0.49,.85,.15,[14400,28800])

#DataGenerator([1,2],1000,0.47,0.85,0.15,[14400,154647],False)
print("My program took", time.time() - start_time, "to run")