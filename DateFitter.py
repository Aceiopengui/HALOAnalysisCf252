import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from itertools import zip_longest
import Double_Count_Distributor as MultiCounter
import scipy.optimize
import numdifftools
import DataGenerator_ColinMethod as DataGenerator


def FinderYear(guess):
    L=0
    #[year0,initialamount,efficiency]
    years=[guess[0],guess[0]+0.128,guess[0]+0.246,guess[0]+0.323,guess[0]+5.4817]
    #years=[guess[0],guess[0]+1,guess[0]+2,guess[0]+3,guess[0]+4]
    _initialAmount=guess[1]
    _Eff=guess[2]
    #_Eff=0.47
    realRunSums=[236413,231891,228186,225459]
    #if _initialAmount<=1 or years[0]<=0 or years[1]-years[0]>=1.1 or years[1]-years[0]<=0.9 or years[2]-years[0]>=2.1 or years[2]-years[0]<=1.9 or years[3]-years[0]>=3.1 or years[3]-years[0]<=2.9 or years[0]>=50 or _Eff>=0.5:
       # print("One of the parameters is invalid")
       # return(10000000000000000000000000000000)
    if years[0]<=0 or _initialAmount<=1 or np.abs(years[1]-years[0])-0.128>=0.011 or np.abs(years[2]-years[1])-0.118>=0.011 or np.abs(years[3]-years[2])-0.077>=0.011 or np.abs(years[4]-years[3])-5.1587>=0.011:
         return(10000000000000000000000000000000)
     
    _tempResult=DataGenerator.DataGenerator(years,_initialAmount,_Eff,Cf252,Cf250,lengthOfRuns,True)
    print(_tempResult,"TEMPRESULT")
    guessVector=[0]*(len(data[0]))
    for n in range(len(_tempResult)):
        for n2 in range(len(data[n])-1):
            guessVector[n2]=_tempResult[n][n2]
            
            if guessVector[n2]<0:
                print("Guess Vector is below 0! Error!")
                return(10000000000000000000000000000000000000)
            if data[n][n2]==0:
                #changed guessVector[n] to guessVector[n2] - will test results
                L+=guessVector[n2]
            elif guessVector[n2]<=0:
                L+=0
            else:
                L+=np.abs(2*(data[n][n2]*np.log((data[n][n2]/guessVector[n2])) + guessVector[n2] - data[n][n2]))

    
    #print(guess)
    #print(L)

    if L<=0.1:
        
        print(years,"YEARS AT HYPER LOW L")
    
    return(L)


#Current mystery: The fit works great when constraints are put in on years with respect to eachother and not at all when there are no such constraints

def DateFitter(Data,Cf252Proportion,Cf250Proportion,guess,runLengths):
    global Cf252
    global Cf250
    global lengthOfRuns
    global data
    Cf252=Cf252Proportion
    Cf250=Cf250Proportion
    lengthOfRuns=runLengths
    
    data=Data
    #results=FinderYear([12,13,14,15,10000000,0.47])
    #results=scipy.optimize.minimize(FinderYear,guess,method='Nelder-Mead',options={'maxfev':70000,'disp':True})
    #results=FinderYear(guess)
    
    results=scipy.optimize.differential_evolution(FinderYear,bounds=[(0,100),(1,1000000000),(0.35,0.55)],strategy='currenttobest1exp')#,x0=guess)
    #guess[0],guess[0]+0.128,guess[0]+0.246,guess[0]+0.323
    #results=scipy.optimize.minimize(FinderYear, guess,method='L-BFGS-B')
    print(results,"results")
    #values=[results['x'][0],results['x'][0]+0.077,results['x'][0]+0.195,results['x'][0]+0.323,results['x'][4],results['x'][5]]
    #values=[results['x'][0],results['x'][0]+0.128,results['x'][0]+0.246,results['x'][0]+0.323,results['x'][0]+5.4817,results['x'][1],results['x'][2]]
    values=results['x']
    #step=values*0.01
    Hfun = numdifftools.Hessian(FinderYear, step=[0.1,10000000,0.001], full_output="true")
    hessian_ndt, info = Hfun(values)
    print(hessian_ndt,"Hessian_ndt")
    print(np.diag(np.linalg.inv(hessian_ndt)),"Hfun")
    print(Hfun(values),"hessian")
    se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
    print(se,"Fit uncertainty (SE)")
    # print(values,"values")
    #Hfun=numdifftools.Hessian(FinderYear,full_output="true")
    #hessian_ndt,info=Hfun(values)
   # print(hessian_ndt,"Hessian_ndt")
    #print(np.diag(np.linalg.pinv(hessian_ndt)),"Hfun")
   # print(Hfun(values),"Hessian")
    #se=np.sqrt(np.diag(np.linalg.pinv(hessian_ndt)))
    #print("Fit Uncertainty: "+str(se))
    #print(values,"VALUES")
    
    #runLengthArray=[14400,14400,14400,14400]
    #atest=[]

    #print(atest,"atest")
    #realRunSums=[236413,231891,228186,225459]
    
    
    print("============================")
    print(values[0],"\n +/- ",se[0]," Age of the First Run")
    print((float(values[0])+0.128),"\n +/-",se[0],"Age of the Second Run")
    print(float(values[0])+0.246,"\n +/-",se[0],"Age of the Third Run")
    print(float(values[0])+0.323,"\n +/-",se[0],"Age of the Fourth Run")
    print(float(values[0])+5.4817,"\n +/-",se[0],"Age of the Sample Today")
    print(values[1],"\n +/- ",se[1]," Initial Amount")
    print(values[2],"\n +/- ",se[2]," Efficiency")
    #print(values[3])#," +/- ",se[3]," Age of the fourth run")
    #print(values[4])#," +/- ",se[4]," Age of the fifth run")
    #print(values[5])#," +/- ",se[5]," Mass Number")
    #print(values[6])#," +/- ",se[6]," Efficiency")

    print("============================")

#print((-np.log(2)*(17.255-16.932))/(np.log(atest[3]/atest[0])),"Slope")

#print((-np.log(2)*(17.255-16.932))/(np.log(realRunSums[3]/realRunSums[0])),"Target Slope")

   # #bin1=[DataGenerator([values[0],values[1],values[2],values[3]],values[4],values[5],0.85,0.15,[14400,14400,14400,14400])]
    fig=plt.figure()
    ax2=fig.add_subplot(2,2,2)


    ##xForRealSums=[values[0],values[1],values[2],values[3]]
   ## plt.plot(xForRealSums,bin1[0])
    #errorSums=[np.sqrt(realRunSums[0]),np.sqrt(realRunSums[1]),np.sqrt(realRunSums[2]),np.sqrt(realRunSums[3])]
    fig=plt.figure()
    print("___________________________")
   ## plt.errorbar(xForRealSums,realRunSums,yerr=errorSums,fmt='o',label="From Data")
    #plt.plot(xForRealSums,atest,color='red',label="From Generator")
    plt.title("Integrated fissions over time")
    plt.ylabel("Integrated number of fissions")
    plt.xlabel("Time (Years)")
    plt.legend()
    plt.show()
    print("_______________________________")
    

# =============================================================================
# realData0,realData1,realData2,realData3,realData4=[],[],[],[],[]
# #realData0.append(np.genfromtxt('5664.csv',delimiter=',')),realData1.append(np.genfromtxt('5719.csv',delimiter=',')),realData2.append(np.genfromtxt('5768.csv',delimiter=',')),realData3.append(np.genfromtxt('5807.csv',delimiter=','))
# realData0.append(np.genfromtxt('TestDataFissions12.csv',delimiter=',')),realData1.append(np.genfromtxt('TestDataFissions13.csv',delimiter=',')),realData2.append(np.genfromtxt('TestDataFissions14.csv',delimiter=',')),realData3.append(np.genfromtxt('TestDataFissions15.csv',delimiter=',')),realData4.append(np.genfromtxt('TestDataFissions16.csv',delimiter=','))
#     
# datae=([realData0,realData1,realData2,realData3,realData4])
# data=np.concatenate(datae)
# =============================================================================
#originalYear=36.623476
#originalYear=17
# Years=[originalYear,originalYear+0.128,originalYear+0.246,originalYear+0.323]
#Eff=0.5

# =============================================================================
# yearGuess=[12]#,13,14,15]
# initialAmountGuess=12000000
# EffGuess=0.48
# Cf252=0.85
# Cf250=0.15
# guess=[yearGuess[0],initialAmountGuess,EffGuess]
# =============================================================================

#DateFitter(data,Cf252,Cf250,guess)