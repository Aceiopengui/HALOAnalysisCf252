import numpy as np
import matplotlib.pyplot as plt
import math
import numdifftools
import scipy.optimize
from scipy.special import comb
from itertools import zip_longest
#import Data_Generator_Complete as DataGenerator
#import Debugging_Data_Generator as GenerateData
#from DataGenerator_SingleMethod import CurrentFissionProportionComputer,ComputeEfficiency,DistributionAdder,FissionDater,BackgroundHandler,DistributionNormalizer,CurrentFissionComputer,SingleCounts,DoubleCounts,TripleCounts,DoubleCountScaler,MultiCountDistributor,FinalDistributionComputer,FissionProportions,InitialFissionComputer,ConvertNeutronsToMassProportions
from DataGenerator_ColinMethod import FissionRate,FissionProportions,CurrentFissionComputer,EarlyDistComputer,DistComputer,SingleProportion,DoubleProportion,TripleProportion,DistributionAdder,ComputeEfficiency2,BackgroundHandler,ConvertNeutronsToMassProportions,PropogateErrorMass
import Double_Count_Distributor as MultiCounter

def ProportionFitter(data,yearOfInterest,runLength):
    
    #Efficiency=0.5
    Cf252InitialProportion=1
    Cf250InitialProportion=1
    #initial252ProportionGuess,initial250ProportionGuess=DataGenerator.DataGenerator(yearOfInterest,initialNeutrons,Efficiency,Cf252InitialProportion,Cf250InitialProportion)
    
    dataFromGenerator=[]
    dataFromGenerator1=[]
    dataFromGenerator2=[]
    dataFromGenerator3=[]
    dataFromGenerator4=[]
    dataFromGenerator5=[]
    dataFromGenerator6=[]
    dataFromGenerator7=[]
    dataFromGenerator8=[]
    dataFromGenerator9=[]
    dataFromGenerator10=[]
    dataFromGenerator11=[]
    dataFromGenerator12=[]
    dataFromGenerator13=[]
    dataFromGenerator14=[]
    dataFromGenerator15=[]
    dataFromGenerator16=[]
    dataFromGenerator17=[]
    dataFromGenerator18=[]
    dataFromGenerator19=[]
    dataFromGenerator20=[]
    dataFromGenerator21=[]
    dataFromGenerator23=[]
    dataFromGenerator31=[]
    dataFromGenerator32=[]
    dataFromGenerator33=[]
    dataFromGenerator34=[]
    dataFromGenerator35=[]
    dataFromGenerator36=[]
    dataFromGenerator55=[]
    dataFromGenerator57=[]
    dataFromGenerator504=[]
    dataFromGenerator221=[]
    dataFromGenerator2211=[]
    dataFromGenerator2212=[]
    dataFromGenerator2213=[]
    dataFromGenerator2214=[]
    dataFromGenerator2215=[]
    dataFromGenerator222=[]
    dataFromGenerator223=[]
    dataFromGenerator227=[]
    dataWithNewFit1,dataWithNewFit2,dataWithNewFit3,dataWithNewFit4,dataWithNewFit5,dataWithNewFit6=[],[],[],[],[],[]
    dataFromGeneratorsame1,dataFromGeneratorsame2,dataFromGeneratorsame3,dataFromGeneratorsame4,dataFromGeneratorsame5,dataFromGeneratorsame6,dataFromGeneratorsame7,dataFromGeneratorsame8=[],[],[],[],[],[],[],[]
    fulldata,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10=[],[],[],[],[],[],[],[],[],[],[]
    
    
    
    
    #Toggle if we are using the generated data or the real data interpreted by Colin's Code
    realData=0
    
    if realData==1:
        #Real data runs 5568-5807 taken in "CT-33-10 inserted to center position (148cm)" "	Calibration, Front Shielding, Source in HALO"
        #yearOfInterest=[16.5479,16.778,16.855,16.932,17.060,17.255] #Max's runs (0-4) and Run 5384 of Colin's Runs
        #yearOfInterest=[16.778,16.855,16.932,17.060,17.255,16.548,17.178,16.449] #Max's runs (0-4) and Run 5384 of Colin's Runs
    
        yearOfInterest=[17.400,17.477,17.595,17.723]
        #yearOfInterest=[16.5479,16.5479,16.5479]#Colin's Runs (5-9)
    
        time=np.linspace(0,np.round(max(yearOfInterest)),1000)
        dataFromGenerator.append(np.genfromtxt('5568.csv',delimiter=','))
        dataFromGenerator1.append(np.genfromtxt('5607.csv',delimiter=','))
        dataFromGenerator2.append(np.genfromtxt('5664.csv',delimiter=','))
        dataFromGenerator3.append(np.genfromtxt('5719.csv',delimiter=','))
        dataFromGenerator4.append(np.genfromtxt('5807.csv',delimiter=','))
        dataFromGenerator5.append(np.genfromtxt('5380.csv',delimiter=','))#Class 1 (Low Eff) (Eff about ~12.5%)
        dataFromGenerator6.append(np.genfromtxt('5382.csv',delimiter=','))#Class 1 (Low Eff)
        dataFromGenerator7.append(np.genfromtxt('5384.csv',delimiter=','))#Class 3 (High Eff) (Works with Max's Runs, same Eff, 39~%) 1 hour 20 mins
        dataFromGenerator8.append(np.genfromtxt('5386.csv',delimiter=','))#Class 1 (Low Eff)
        dataFromGenerator9.append(np.genfromtxt('5388.csv',delimiter=','))#Class 2 (Med Eff) (Eff about ~19%)
        dataFromGenerator10.append(np.genfromtxt('4860.csv',delimiter=','))#11 minute run shown on Colin's Thesis, top of page 46
        dataFromGenerator11.append(np.genfromtxt('5768.csv',delimiter=','))#Colin's - middle of detector 4 hours
        dataFromGenerator12.append(np.genfromtxt('5263.csv',delimiter=','))#Colin's - middle of detector 2 hours
        dataFromGenerator221.append(np.genfromtxt('TestDataFissions22.1.csv',delimiter=','))#Test data generated to represent today
        dataFromGenerator222.append(np.genfromtxt('TestDataFissions22.2.csv',delimiter=','))#Test data generated to represent 0.1 years in the future
        dataFromGenerator223.append(np.genfromtxt('TestDataFissions22.3.csv',delimiter=','))#Test data generated to represent 0.2 years in the future
        dataFromGenerator2211.append(np.genfromtxt('TestDataFissions22.11.csv',delimiter=','))#Test data generated to represent 0.2 years in the future
        dataFromGenerator2212.append(np.genfromtxt('TestDataFissions22.12.csv',delimiter=','))#Test data generated to represent 0.2 years in the future
        dataFromGenerator2213.append(np.genfromtxt('TestDataFissions22.13.csv',delimiter=','))#Test data generated to represent 0.2 years in the future
        dataFromGenerator2214.append(np.genfromtxt('TestDataFissions22.14.csv',delimiter=','))#Test data generated to represent 0.2 years in the future
        dataFromGenerator2215.append(np.genfromtxt('TestDataFissions22.15.csv',delimiter=','))#Test data generated to represent 0.2 years in the future
        dataWithNewFit5.append(np.genfromtxt('TestDataFissions38.69076.csv',delimiter=','))
        dataWithNewFit6.append(np.genfromtxt('TestDataFissions22.723.csv',delimiter=','))
        datae=([dataFromGenerator2,dataFromGenerator3,dataFromGenerator4,dataFromGenerator11])
        runLength=[14400,14400,14400,14400]
        print((dataFromGenerator),"LENGTH")
        
        print(len(runLength),len(datae),len(yearOfInterest))
        data=np.concatenate(datae)
        
# =============================================================================
#     if realData==0:
#     
#         ##yearOfInterest=[16.932,17.06,17.178,17.255,22.255]#,22.255,22.34]
#         yearOfInterest=[12,13,14,15]
#         time=np.linspace(0,np.round(max(yearOfInterest)),1000)
#     
#         for i in range(len(yearOfInterest)):
#             fulldata.append(np.genfromtxt(('TestDataFissions'+str(yearOfInterest[i])+'.csv'),delimiter=','))
#     
#         runLength=[14400,14400,14400,144000]#,259200]
#        
#         ##datae=np.array([dataFromGeneratorsame3,dataFromGeneratorsame4,dataFromGeneratorsame7,dataFromGeneratorsame5,dataWithNewFit5])
#     
#         print(len(fulldata),"testlength")
#         print(len(yearOfInterest),"testyearlength")
#         data=np.array(fulldata)
# =============================================================================
    
    
    
    
    time=np.linspace(0,np.round(max(yearOfInterest)),1000)    
    decayConst252=np.log(2)/2.65
    decayConst250=np.log(2)/13.08
    initialMass252,initialMass250=1,1
    #runLength=14402#Length of run in seconds
    
    channelProportion252Alpha=0.96908
    channelProportion252Fission=1-channelProportion252Alpha
    channelProportion250Alpha=0.99923
    channelProportion250Fission=1-channelProportion250Alpha
    
    #This is for run 5557
    #protoBackgroundEvents=[5806,30,10,1]
    protoBackgroundEvents=[14400/60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    backgroundRunLength=14400
    
    timeWindow=0.002#Time window in seconds,  might be increased to 0.002 in the future
    multi252=3.735
    multi250=3.50
    
    initialNeutronProportion252=(decayConst252*Cf252InitialProportion*channelProportion252Fission*multi252)/((decayConst252*Cf252InitialProportion*channelProportion252Fission*multi252+decayConst250*Cf250InitialProportion*channelProportion250Fission*multi250))
    initialNeutronProportion250=(decayConst250*Cf250InitialProportion*channelProportion250Fission*multi250)/((decayConst250*Cf250InitialProportion*channelProportion250Fission*multi250+decayConst252*Cf252InitialProportion*channelProportion252Fission*multi252))
    
    
    cf252MultiNoNorm=[0.026,0.1267,0.2734,0.3039,0.1848,0.0657,0.0154,0.0020,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    cf250NultiNoNorm=[0.03594,0.16771,0.29635,0.30104,0.14531,0.04740,0.00417,0.00208,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    Cf252Multi=[x / sum(cf252MultiNoNorm) for x in cf252MultiNoNorm]
    Cf250Multi=[x / sum(cf250NultiNoNorm) for x in cf250NultiNoNorm]
    current252distlist,current250distlist=[0]*(len(data1)-1),[0]*(len(data1)-1)
    timeToggle=0
    
    poptfermi = np.genfromtxt('FermiParameters.csv',delimiter=',')
    
    for n3 in range(len(time)):
        fitFermiFunction=poptfermi[0]/(poptfermi[2]+poptfermi[3]*np.exp(poptfermi[5]*(-np.array(time)+poptfermi[4])))+poptfermi[1]
    
    
    
    global distforgraph252
    #distforgraph252=np.array([[0]*len((data[0])),[0]*len((data[0])),[0]*len((data[0])),[0]*len(data[0])])
    distforgraph252=np.zeros((len(yearOfInterest),len(data[0])))
    global distforgraph250
    #distforgraph250=np.array([[0]*len((data[0])),[0]*len((data[0])),[0]*len((data[0])),[0]*len(data[0])])
    distforgraph250=np.zeros((len(yearOfInterest),len(data[0])))
    global singdistforgraph
    global dubdistforgraph
    global tripdistforgraph
    global protosingdistforgraph
    global protodubdistforgraph
    global prototripdistforgraph
    protosingdistforgraph,protodubdistforgraph,prototripdistforgraph=[],[],[]
    singdistforgraph,dubdistforgraph,tripdistforgraph=np.zeros((len(yearOfInterest),len(data[0]))),np.zeros((len(yearOfInterest),len(data[0]))),np.zeros((len(yearOfInterest),len(data[0])))
    
    def MLEfunction(params):
        amount252, amount250, Efficiency= params[0], params[1], params[2]
        guessVector = [0]*(len(data[0])-1)
        L = 0
       # print(amount252,"amount252")
       # print(amount250,"amount250")
        #print(Efficiency,"Efficiency")
    
        if amount252<=0 or amount250<=0 or Efficiency>1 or Efficiency<0:
            #print("pitfall 1")
            return(1000000000000000)
        
        #Put loop that evaluates these for each year
        for n2 in range(len(yearOfInterest)):
            runLengthFactor=runLength[n2]/runLength[0]
            backGroundToData=runLength[n2]/backgroundRunLength
            backgroundEvents=[np.round(x*backGroundToData) for x in protoBackgroundEvents]
            backgroundRate=[x / backgroundRunLength for x in backgroundEvents]
            currentFissions252=CurrentFissionComputer(amount252,decayConst252,yearOfInterest[n2],runLengthFactor)
            currentFissions250=CurrentFissionComputer(amount250,decayConst250,yearOfInterest[n2],runLengthFactor)
            totalCurrentFissions=currentFissions252+currentFissions250
            preEffDistTotal=DistComputer(currentFissions252,currentFissions250,Cf252Multi,Cf250Multi)
            colinFactorSingle=np.exp(-2*totalCurrentFissions*timeWindow/runLength[n2])
            colinFactorDouble=3*np.exp(-totalCurrentFissions*timeWindow/runLength[n2])-2*np.exp(-2*totalCurrentFissions*timeWindow/runLength[n2])-1
            colinFactorTriple=(1-np.exp(-totalCurrentFissions*timeWindow/runLength[n2]))**2
    
            preEffCurrentFissions252ForGraph=[x*currentFissions252 for x in Cf252Multi]
            preEffCurrentFissions250ForGraph=[x*currentFissions250 for x in Cf250Multi]
            
            currentFissions252ForGraph=ComputeEfficiency2(preEffCurrentFissions252ForGraph,Efficiency)
            currentFissions250ForGraph=ComputeEfficiency2(preEffCurrentFissions250ForGraph,Efficiency)
    
    
            singleCountProportion=SingleProportion(preEffDistTotal,colinFactorSingle)
            doubleCountProportion,doubleForTriple=DoubleProportion(preEffDistTotal,colinFactorDouble,runLengthFactor)
            tripleCountProportion=TripleProportion(preEffDistTotal,colinFactorTriple,doubleForTriple,runLengthFactor)
            protosingdistforgraph=[totalCurrentFissions*x for x in ComputeEfficiency2(singleCountProportion,Efficiency)]
            protodubdistforgraph=[totalCurrentFissions*x for x in ComputeEfficiency2(doubleCountProportion,Efficiency)]
            prototripdistforgraph=[totalCurrentFissions*x for x in ComputeEfficiency2(tripleCountProportion,Efficiency)]
            fullDistributionBeforeEfficiency=DistributionAdder(singleCountProportion,doubleCountProportion,tripleCountProportion)
            #fullDistributionBeforeEfficiency=[x/sum(protofullDistributionBeforeEfficiency) for x in protofullDistributionBeforeEfficiency]
            fullDistributionPreBackground=ComputeEfficiency2(fullDistributionBeforeEfficiency,Efficiency)
            #normalizedDistribution=[x/sum(fullDistributionPreBackground) for x in fullDistributionPreBackground]
            a=[x*totalCurrentFissions for x in fullDistributionPreBackground]
            b=BackgroundHandler(a,backgroundEvents,runLength[n2])
    #TO DO: FIX MLE FUNCTION WITH COLIN'S CODE. Go over each part and make sure it's generating the same sort of distribution.
    
            for i in range(len(data[n2])-1):
     
                singdistforgraph[n2][i]=protosingdistforgraph[i]
                dubdistforgraph[n2][i]=protodubdistforgraph[i]
                tripdistforgraph[n2][i]=prototripdistforgraph[i]
                distforgraph252[n2][i]=currentFissions252ForGraph[i]
                distforgraph250[n2][i]=currentFissions250ForGraph[i]
    
                
                guessVector[i]=b[i]
    
                if guessVector[i]<=0:
                    return(10000000000000000000)
        
                if data[n2][i]==0:
                    L+=guessVector[i]
                elif guessVector[i]<=0:
                    L+=0
                else:
                    
                    L += np.abs(2*(data[n2][i]*math.log((data[n2][i]/guessVector[i])) + guessVector[i] - data[n2][i]))
        #end above defined loop
    
    
        
        return(L)
    
    
    
    bins=np.arange(1,len(data[0])+1)
    
    barwidth=0.25
    if timeToggle==0:
        
    
        #POSSIBLE PROBLEM I am using the mass proportions with the neutron amounts. This can be solved with some math similmiar to how I did in the data generation code
        #----CAN'T PUT A 2D ARRAY INTO THE SCIPY MINIMIZE FUNCTION----
        #THE PROBLEM IS THAT THIS GUESS VECTOR IS GIVING THE MLE FUNCITON AN INITIAL FISSIONS AMOUNT, BUT THE MLE FUNCTION IS USING MY DATA GENERATION CODE WHICH TAKES IN THE CURRENT FISSIONS AMOUNT
        guess=np.array([.8*sum(data[0]),.2*sum(data[0]),0.6])
        print(guess,"guess")
        results = scipy.optimize.minimize(MLEfunction, guess, method = 'Nelder-Mead', options = {'maxfev':4000,'disp': True})
    
    
    
        #Error Estimation
        
    
    
        for n3 in range(len(yearOfInterest)):
            values=results['x']
            totalCurrentFissions=values[0]+values[1]
            Hfun = numdifftools.Hessian(MLEfunction, full_output="true")
            hessian_ndt, info = Hfun(values)
            print(hessian_ndt,"Hessian_ndt")
            print(np.diag(np.linalg.inv(hessian_ndt)),"Hfun")
            print(Hfun(values),"hessian")
            se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
            print("Fit Uncertainty: " + str(se))
            massprop252,massprop250=ConvertNeutronsToMassProportions(values[0],values[1],decayConst252,decayConst250,channelProportion252Fission,channelProportion250Fission,yearOfInterest[n3])
            
            #FIX THIS NEXT TIME
            masserror252,masserror250=ConvertNeutronsToMassProportions(values[0]+se[0],values[1]+se[1],decayConst252,decayConst250,channelProportion252Fission,channelProportion250Fission,yearOfInterest[n3])
            downmasserror252,downmasserror250=ConvertNeutronsToMassProportions(values[0]-se[0],values[1]-se[1],decayConst252,decayConst250,channelProportion252Fission,channelProportion250Fission,yearOfInterest[n3])
            wisemasserror252,wisemasserror250=PropogateErrorMass(values[0],values[1],se[0],se[1],decayConst252,decayConst250,channelProportion252Fission,channelProportion250Fission,yearOfInterest[n3])
            
            
            print(np.abs(masserror252-values[0]),np.abs(masserror250-values[1]),"masserror")
            print(np.abs(downmasserror252-values[0]),np.abs(masserror250-values[1]),"downmasserror")
            
            error252denominator=se[0]+se[1]
            error252=((values[0])/(values[0]+values[1]))*np.sqrt(((se[0]/values[0])**2+(error252denominator/(values[0]+values[1]))**2))
            error250=((values[1])/(values[0]+values[1]))*np.sqrt(((se[1]/values[1])**2+(error252denominator/(values[0]+values[1]))**2))
            
            print("___________________________________")
            print("Year: ",yearOfInterest[n3])
            print(values[0],"+/-",se[0],"Amount of Cf252")
            print(values[1],"+/-",se[1],"Amount of Cf250")
            print("--------")
    
            print((massprop252),"+/-",str(wisemasserror252),"Proportion of Cf252 Mass")
            print((massprop250),"+/-",str(wisemasserror250),"Percentage of Cf250 Mass")
            print(100*values[2],"+/-",str(100*se[2]),"Efficiency (%)")
            print("___________________________________")
    
    
    #n4=0
    #print(distforgraph252,"DIST FOR GRAPH 252")
    for n4 in range(len(yearOfInterest)):
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.set_yscale('log')
        combinedList=[a+b for a,b in zip(distforgraph252[n4],distforgraph250[n4])]
        errorList=np.sqrt(data[n4])
        #achart=plt.plot(bins,distforgraph252[n4],label="Cf252 Fissions")
        #bchart=plt.plot(bins,distforgraph250[n4],label="Cf250 Fissions")
        #summedchart=plt.plot(bins,combinedList,label="Summed Graph")
    
        plt.xticks(bins,bins)
        fulldist=[sum(x) for x in zip_longest(singdistforgraph[n4],dubdistforgraph[n4],tripdistforgraph[n4])]
        plt.plot(bins,np.round(singdistforgraph[n4]),label="Single counts")
        plt.plot(bins,np.round(dubdistforgraph[n4]),label="Double counts")
        plt.plot(bins,np.round(tripdistforgraph[n4]),label="Triple counts")    
        plt.plot(bins,protoBackgroundEvents,label="Background Events")
        plt.plot(bins,np.round(fulldist),label="Sum of Fitted Data")
        plt.plot(bins,np.round(distforgraph252[n4]),label="Contributions from Cf252")
        plt.plot(bins,np.round(distforgraph250[n4]),label="Contributions from Cf250")
        print(len(bins),len(data[n4]),"Length test")
        plt.scatter(bins,data[n4],label="Original Data",marker="x")
        #plt.errorbar(bins,data[n4],label="Input Data (Fissions)",marker='x')
        #datainput=plt.scatter(bins,data[n4],label="Input Data (Fissions)",marker='x')
        plt.errorbar(bins,data[n4],xerr=0,yerr=50*errorList,fmt='+',capsize=5,label="Erorr Bars on Data")
        plt.title("Sample Age: "+str(yearOfInterest[n4]))
        #_______________________
        #Questions for Tom: Is efficiency an adequate reason for why double and triple counts can have events in bin 1? Without efficiency they should not but with any <100% efficiency they do.
        #Error bars look gross - are they valid?
        #"Contributions from" not appearing in the full graph, just the beginning (Only a part of single counts, really - single counts maps on very closely to the Contributions from 252 on a well fit dataset.).
        #_______________________
        returnedlist=[]
        for (item1,item2) in zip(distforgraph252[n4],distforgraph250[n4]):
            returnedlist.append(item1+item2)
        #print(returnedlist,"Back to distribution")
        
        plt.legend()
        plt.xlim(0,15)
        plt.ylim(0.5,1.2*max(data[n4]))
        plt.show()
        
        
    #MLEfunction([22971026,28974,.47])
