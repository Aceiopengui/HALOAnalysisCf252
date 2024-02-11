import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
import CfCmDataGeneratorT0
import time
from scipy import interpolate



#Start the timer that will tell us how long the program took to run
start_time=time.time()
#Parameters of the runs taken in 1996 and 2011
oldRunsAges,oldRunsRates,oldRunsErrors=[1.3589,16.063],[8965,215.88],[269,6.48]
#Physical Parameters
decayConst252,decayConst250=np.log(2)/2.65,np.log(2)/13.08
chanAlpha252,chanAlpha250=0.96908,0.99923
chanFiss252,chanFiss250=1-chanAlpha252,1-chanAlpha250
multi252,multi250=3.735,3.50

#Define the Chi Square with Error for including the older runs mentioned above
def ChiSquareWithError(obs,exp,error):
    chisq=0
    chisq+=((obs-exp)**2)/(error**2)
    return chisq

#The Likelihood fuction
def FinderYear(guess):
    L=0
    #Grabbing parameters. Since mass ratio between Cf252/Cf250 is passed in and the data generator takes mass as a percentage of its total, a conversion must be made.
    #years=[guess[0],guess[0]+0.3825,guess[0]+0.5123,guess[0]+0.6301,guess[0]+0.7068,guess[0]+5.8630]
    years=[guess[0],guess[0]+0.3835,guess[0]+0.5123,guess[0]+0.6329,guess[0]+0.7068,guess[0]+5.8630,guess[0]+6.3835]
    _initialAmount=guess[1]
    _Eff=guess[2]
    Cf250Ratio=guess[3]
    mysteryIsotope=guess[4]
    Cf252=1/(1+Cf250Ratio)
    Cf250=1-Cf252
    
    #If at any point in the fit any parameters become unphysical, the fitter is told  "no".
    if Cf252<=0 or Cf250<0 or np.abs(1-Cf252-Cf250)>=0.0001 or mysteryIsotope<=0 or _initialAmount<=0:
        return(1000000000000000000000000000000000000000000000000)
    #Generates data with the given parameters
    _tempResult=CfCmDataGeneratorT0.DataGenerator(years,_initialAmount,_Eff,Cf252,Cf250,mysteryIsotope,runLengths,backgroundRate,False,False)
    #Estimates the shape of the spectrum from the runs taken not at SNOLAB
    _oldTempResult=CfCmDataGeneratorT0.DataGenerator(oldRunsAges,_initialAmount,_Eff,Cf252,Cf250,mysteryIsotope,[14402,14402],[0,0],False,False)
    totalSumArray=[]
    #Finds the total amount of neutrons expected at the time each of those runs were taken to be fed into the Chi Square function
    for q in range(len(_oldTempResult)):
        totalSum=0
        for q2 in range(len(_oldTempResult[0])):
            totalSum+=_oldTempResult[q][q2]*(q2+1)
        totalSumArray.append(totalSum/runLengths[q])
    #Computes the Chi Square error of the runs not taken at SNOLAB so they can be included in this fit
    ORNLRun1=ChiSquareWithError(oldRunsRates[0],totalSumArray[0],oldRunsErrors[0])/2
    ORNLRun2=ChiSquareWithError(oldRunsRates[1],totalSumArray[1],oldRunsErrors[1])/2
    #Biases the fit in favor of the Cf252/Cf250 ratio found in the ICPMS report provided by Frontier
    cf250Fixer=ChiSquareWithError(1/8.164,CfCmDataGeneratorT0.Cf250MassFractionGrabber(Cf252,Cf250,decayConst252,decayConst250,chanFiss252,chanFiss250,years[0]),(1/8.164)*1.25*0.05)/2
    #Instantiates the list that will be compared with the data to determine the log likelihood
    guessVector=[0]*len(data[0])
    
    for n in range(len (_tempResult)):
        for n2 in range(len(data[n])-1):
            guessVector[n2]=_tempResult[n][n2]
            #Doesn't allow any neutron bins to be represented as a negative amount of neutrons
            if guessVector[n2]<0:
                return(1000000000000000000000000000000000000000000000)
            #Allows exceptions. Firstly (ASK TOM). Secondly, if the data generator is suggesting 0, then 0 is added to the log likelihood
            if data[n][n2]==0:
                L+=guessVector[n2]
            elif guessVector[n2]<=0:
                L+=0
            else:
                L+=np.abs(((data[n][n2]*np.log((data[n][n2]/guessVector[n2]))+guessVector[n2]-data[n][n2])))
    #Before returning the parameter to be minimized, allow it to consider the non-SNOLAB runs and the ICPMS report computed above.
    L+=ORNLRun1+ORNLRun2+cf250Fixer
    return(L)
        
def ErrorComputerEffRatio(values):
    print("Initializing Efficiency and Ratio Error Analysis....")
    ErrorTime=time.time()
    resolution=16
    errorEffRatio=[]
    xaxis=np.linspace(values[3]-0.004,values[3]+0.004,resolution)
    yaxis=np.linspace(values[2]-0.0004,values[2]+0.0004,resolution)
    minimumL=FinderYear([values[0],values[1],values[2],values[3],values[4]])
    index=0
    
    for indexForLoop1 in range(resolution):
        print(indexForLoop1)
        effToUse=yaxis[indexForLoop1]
        for indexForLoop2 in range(resolution):
            #print("__________________________")
            print(index+1,"out of ", len(xaxis)**2)
            ratioToUse=xaxis[indexForLoop2]
            guess=[values[0],values[1],effToUse,ratioToUse,values[4]]
            errorResults=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(21.4603,21.4603),(1000000,100000000),(effToUse,effToUse),(ratioToUse,ratioToUse),(0.,100.)])
            errorValues=errorResults['x']
            currentL=FinderYear([errorValues[0],errorValues[1],effToUse,ratioToUse,errorValues[4]])
            errorEffRatio.append(currentL)
            index=index+1
    Z1=np.array(errorEffRatio)
    Z2=Z1.reshape(len(xaxis),len(yaxis))
    print(Z2,"Z2 in fit 1")
    fig,ax=plt.subplots(dpi=250)
    X1,X2=np.meshgrid(xaxis,yaxis)
    levels=np.linspace(minimumL,minimumL+2,10)
    cp=ax.contourf(X1,X2,Z2,levels=levels)
    fig.colorbar(cp)
    plt.title("Error Landscape")
    cp2=ax.contour(X1,X2,Z2,[FinderYear([values[0],values[1],values[2],values[3],values[4]])+0.5],colors='red')
    for item in cp2.collections:
        for i in item.get_paths():
            v=i.vertices
            vx=v[:,0]
            vy=v[:,1]
            print(vx,vy,"allegedly xy points on the l=min+0.5 line")
            print("From", min(vx)," to ",max(vx)," in the proportion axis:",values[3]," (+/-) ", np.abs(max(vx)-min(vx))/2)
            print("From", min(vy)," to ",max(vy)," in the efficiency axis:",values[2]," (+/-) ", np.abs(max(vy)-min(vy))/2)
    plt.xlabel("Initial Ratio Of Cf252/Cf250 Mass")
    plt.ylabel("Efficiency of Detector")
    plt.show()
    print("Error Analysis of the Efficiency and Ratio took", np.abs(ErrorTime-time.time())," seconds to run.")

def ErrorComputerEffMystery(values):
    print("Initializing Efficiency and Mystery Isotope Analysis....")
    ErrorTime=time.time()
    resolution=16
    errorEffMystery=[]
    xaxis=np.linspace(values[4]-0.15,values[4]+0.15,resolution)
    yaxis=np.linspace(values[2]-0.0004,values[2]+0.0004,resolution)
    minimumL=FinderYear([values[0],values[1],values[2],values[3],values[4]])
    index=0
    
    for indexForLoop1 in range(resolution):
        effToUse=yaxis[indexForLoop1]
        for indexForLoop2 in range(resolution):
            #print("__________________________")
            print(index+1,"out of ", len(xaxis)**2)
            mysteryToUse=xaxis[indexForLoop2]
            guess=[values[0],values[1],effToUse,values[3],mysteryToUse]
            errorResults=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(21.4603,21.4603),(1000000,100000000),(effToUse,effToUse),(0.,1000.),(mysteryToUse,mysteryToUse)])
            errorValues=errorResults['x']
            currentL=FinderYear([errorValues[0],errorValues[1],effToUse,errorValues[3],mysteryToUse])
            errorEffMystery.append(currentL)
            index=index+1
    Z1=np.array(errorEffMystery)
    Z2=Z1.reshape(len(xaxis),len(yaxis))
    print(Z2,"Z2")
    fig,ax=plt.subplots(dpi=250)
    X1,X2=np.meshgrid(xaxis,yaxis)
    levels=np.linspace(minimumL,minimumL+2,10)
    cp=ax.contourf(X1,X2,Z2,levels=levels)
    fig.colorbar(cp)
    plt.title("Error Landscape")
    cp2=ax.contour(X1,X2,Z2,[FinderYear([values[0],values[1],values[2],values[3],values[4]])+0.5],colors='red')
    for item in cp2.collections:
        for i in item.get_paths():
            v=i.vertices
            vx=v[:,0]
            vy=v[:,1]
            print(vx,vy,"allegedly xy points on the l=min+0.5 line")
            print("From", min(vx)," to ",max(vx)," in the mystery axis:",values[4]," (+/-) ", np.abs(max(vx)-min(vx))/2)
            print("From", min(vy)," to ",max(vy)," in the efficiency axis:",values[2]," (+/-) ", np.abs(max(vy)-min(vy))/2)
    plt.xlabel("Neutron Rate from Mystery Isotope")
    plt.ylabel("Efficiency of Detector")
    plt.show()
    print("Error Analysis of the Efficiency and Mystery Isotope took", np.abs(ErrorTime-time.time())," seconds to run.")
    
def ErrorComputerAmountMystery(values):
    print("Initializing Amount and Mystery Isotope Analysis....")
    ErrorTime=time.time()
    resolution=16
    errorAmountMystery=[]
    xaxis=np.linspace(values[4]-0.2,values[4]+0.2,resolution)
    yaxis=np.linspace(values[1]-80000,values[1]+80000,resolution)
    minimumL=FinderYear([values[0],values[1],values[2],values[3],values[4]])
    index=0
    
    for indexForLoop1 in range(resolution):
        print(indexForLoop1)
        amountToUse=yaxis[indexForLoop1]
        for indexForLoop2 in range(resolution):
            #print("__________________________")
            print(index+1,"out of ", len(xaxis)**2)
            mysteryToUse=xaxis[indexForLoop2]
            guess=[values[0],amountToUse,values[2],values[3],mysteryToUse]
            errorResults=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(21.4603,21.4603),(amountToUse,amountToUse),(0.,1.),(0.,1000.),(mysteryToUse,mysteryToUse)])
            errorValues=errorResults['x']
            currentL=FinderYear([errorValues[0],amountToUse,errorValues[2],errorValues[3],mysteryToUse])
            errorAmountMystery.append(currentL)
            index=index+1
    Z1=np.array(errorAmountMystery)
    Z2=Z1.reshape(len(xaxis),len(yaxis))
    print(Z2,"Z2 in fit 3")
    fig,ax=plt.subplots(dpi=250)
    X1,X2=np.meshgrid(xaxis,yaxis)
    levels=np.linspace(minimumL,minimumL+2,10)
    cp=ax.contourf(X1,X2,Z2,levels=levels)
    fig.colorbar(cp)
    plt.title("Error Landscape")
    cp2=ax.contour(X1,X2,Z2,[FinderYear([values[0],values[1],values[2],values[3],values[4]])+0.5],colors='red')
    for item in cp2.collections:
        for i in item.get_paths():
            v=i.vertices
            vx=v[:,0]
            vy=v[:,1]
            print(vx,vy,"allegedly xy points on the l=min+0.5 line")
            print("From", min(vx)," to ",max(vx)," in the mystery axis:",values[4]," (+/-) ", np.abs(max(vx)-min(vx))/2)
            print("From", min(vy)," to ",max(vy)," in the efficiency axis:",values[1]," (+/-) ", np.abs(max(vy)-min(vy))/2)
    plt.xlabel("Neutron Rate from Mystery Isotope")
    plt.ylabel("Amount")
    plt.show()
    print("Error Analysis of the Amount and Mystery Isotope took", np.abs(ErrorTime-time.time())," seconds to run.")
    
    
    
def ErrorComputerAmountEfficiency(values):
    print("Initializing Amount and Mystery Isotope Analysis....")
    ErrorTime=time.time()
    resolution=16
    errorAmountEff=[]
    xaxis=np.linspace(values[2]-0.0004,values[2]+0.0004,resolution)
    yaxis=np.linspace(values[1]-80000,values[1]+80000,resolution)
    minimumL=FinderYear([values[0],values[1],values[2],values[3],values[4]])
    index=0
    
    for indexForLoop1 in range(resolution):
        print(indexForLoop1)
        amountToUse=yaxis[indexForLoop1]
        for indexForLoop2 in range(resolution):
            #print("__________________________")
            print(index+1,"out of ", len(xaxis)**2)
            effToUse=xaxis[indexForLoop2]
            guess=[values[0],amountToUse,effToUse,values[3],values[4]]
            errorResults=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(21.4603,21.4603),(amountToUse,amountToUse),(effToUse,effToUse),(0.,1000.),(0,100)])
            errorValues=errorResults['x']
            currentL=FinderYear([errorValues[0],amountToUse,effToUse,errorValues[3],errorValues[4]])
            errorAmountEff.append(currentL)
            index=index+1
    Z1=np.array(errorAmountEff)
    Z2=Z1.reshape(len(xaxis),len(yaxis))
    print(Z2,"Z2 in fit 3")
    fig,ax=plt.subplots(dpi=250)
    X1,X2=np.meshgrid(xaxis,yaxis)
    levels=np.linspace(minimumL,minimumL+2,10)
    cp=ax.contourf(X1,X2,Z2,levels=levels)
    fig.colorbar(cp)
    plt.title("Error Landscape")
    cp2=ax.contour(X1,X2,Z2,[FinderYear([values[0],values[1],values[2],values[3],values[4]])+0.5],colors='red')
    for item in cp2.collections:
        for i in item.get_paths():
            v=i.vertices
            vx=v[:,0]
            vy=v[:,1]
            print(vx,vy,"allegedly xy points on the l=min+0.5 line")
            print("From", min(vx)," to ",max(vx)," in the efficiency axis:",values[2]," (+/-) ", np.abs(max(vx)-min(vx))/2)
            print("From", min(vy)," to ",max(vy)," in the amount axis:",values[1]," (+/-) ", np.abs(max(vy)-min(vy))/2)
    plt.xlabel("Neutron Rate from Efficiency")
    plt.ylabel("Amount")
    plt.show()
    print("Error Analysis of the Amount and Mystery Isotope took", np.abs(ErrorTime-time.time())," seconds to run.")    
    
def ErrorComputerAmountRatio(values):
    print("Initializing Amount and Ratio Analysis....")
    ErrorTime=time.time()
    resolution=16
    errorAmountRatio=[]
    xaxis=np.linspace(values[3]-0.008,values[3]+0.008,resolution)
    yaxis=np.linspace(values[1]-80000,values[1]+80000,resolution)
    minimumL=FinderYear([values[0],values[1],values[2],values[3],values[4]])
    index=0
    
    for indexForLoop1 in range(resolution):
        print(indexForLoop1)
        amountToUse=yaxis[indexForLoop1]
        for indexForLoop2 in range(resolution):
            #print("__________________________")
            print(index+1,"out of ", len(xaxis)**2)
            ratioToUse=xaxis[indexForLoop2]
            guess=[values[0],amountToUse,values[2],ratioToUse,values[4]]
            errorResults=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(21.4603,21.4603),(amountToUse,amountToUse),(0.,1.),(ratioToUse,ratioToUse),(0,100)])
            errorValues=errorResults['x']
            currentL=FinderYear([errorValues[0],amountToUse,errorValues[2],ratioToUse,errorValues[4]])
            errorAmountRatio.append(currentL)
            index=index+1
    Z1=np.array(errorAmountRatio)
    Z2=Z1.reshape(len(xaxis),len(yaxis))
    print(Z2,"Z2 in fit 3")
    fig,ax=plt.subplots(dpi=250)
    X1,X2=np.meshgrid(xaxis,yaxis)
    levels=np.linspace(minimumL,minimumL+2,10)
    cp=ax.contourf(X1,X2,Z2,levels=levels)
    fig.colorbar(cp)
    plt.title("Error Landscape")
    cp2=ax.contour(X1,X2,Z2,[FinderYear([values[0],values[1],values[2],values[3],values[4]])+0.5],colors='red')
    for item in cp2.collections:
        for i in item.get_paths():
            v=i.vertices
            vx=v[:,0]
            vy=v[:,1]
            print(vx,vy,"allegedly xy points on the l=min+0.5 line")
            print("From", min(vx)," to ",max(vx)," in the ratio axis:",values[3]," (+/-) ", np.abs(max(vx)-min(vx))/2)
            print("From", min(vy)," to ",max(vy)," in the amount axis:",values[1]," (+/-) ", np.abs(max(vy)-min(vy))/2)
    plt.xlabel("Ratio of Cf250 to Cf252 at t0")
    plt.ylabel("Amount")
    plt.show()
    print("Error Analysis of the Amount and Cf250/Cf252 Ratio took", np.abs(ErrorTime-time.time())," seconds to run.")  
    
def ErrorComputerRatioMystery(values):
    print("Initializing Isotope and Ratio Analysis....")
    ErrorTime=time.time()
    resolution=16
    errorRatioIsotope=[]
    xaxis=np.linspace(values[3]-0.008,values[3]+0.008,resolution)
    yaxis=np.linspace(values[4]-0.2,values[4]+0.2,resolution)
    minimumL=FinderYear([values[0],values[1],values[2],values[3],values[4]])
    index=0
    
    for indexForLoop1 in range(resolution):
        print(indexForLoop1)
        isotopeToUse=yaxis[indexForLoop1]
        for indexForLoop2 in range(resolution):
            #print("__________________________")
            print(index+1,"out of ", len(xaxis)**2)
            ratioToUse=xaxis[indexForLoop2]
            guess=[values[0],values[1],values[2],ratioToUse,isotopeToUse]
            errorResults=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(21.4603,21.4603),(1000000,100000000),(0.,1.),(ratioToUse,ratioToUse),(isotopeToUse,isotopeToUse)])
            errorValues=errorResults['x']
            currentL=FinderYear([errorValues[0],errorValues[1],errorValues[2],ratioToUse,isotopeToUse])
            errorRatioIsotope.append(currentL)
            index=index+1
    Z1=np.array(errorRatioIsotope)
    Z2=Z1.reshape(len(xaxis),len(yaxis))
    print(Z2,"Z2 in fit 3")
    fig,ax=plt.subplots(dpi=250)
    X1,X2=np.meshgrid(xaxis,yaxis)
    levels=np.linspace(minimumL,minimumL+2,10)
    cp=ax.contourf(X1,X2,Z2,levels=levels)
    fig.colorbar(cp)
    plt.title("Error Landscape")
    cp2=ax.contour(X1,X2,Z2,[FinderYear([values[0],values[1],values[2],values[3],values[4]])+0.5],colors='red')
    for item in cp2.collections:
        for i in item.get_paths():
            v=i.vertices
            vx=v[:,0]
            vy=v[:,1]
            print(vx,vy,"allegedly xy points on the l=min+0.5 line")
            print("From", min(vx)," to ",max(vx)," in the ratio axis:",values[3]," (+/-) ", np.abs(max(vx)-min(vx))/2)
            print("From", min(vy)," to ",max(vy)," in the mystery isotope axis:",values[4]," (+/-) ", np.abs(max(vy)-min(vy))/2)
    plt.xlabel("Ratio of Cf250 to Cf252 at t0")
    plt.ylabel("Neutron Rate from Mystery Isotope")
    plt.show()
    print("Error Analysis of the Mystery Isotope Neutron Rate and Cf250/Cf252 Ratio took", np.abs(ErrorTime-time.time())," seconds to run.")  
    
def ErrorComputer3D(values):

    print("Initializing 3D Error Analysis....")
    ErrorTime=time.time()
    resolution=4
    errorAmountMysteryEff=[]
    preShapedNeutrons=[]
    rangeOfValues=np.linspace(21,35,resolution)
    SDAmount,SDEff,SDProp,SDMystery=[],[],[],[]
    neutronMin,neutronMax=[],[]
    xaxis=np.linspace(values[4]-0.03,values[4]+0.03,resolution)
    yaxis=np.linspace(values[1]-12000,values[1]+12000,resolution)
    zaxis=np.linspace(values[2]-0.0002,values[2]+0.0002,resolution)
    minimumL=FinderYear([values[0],values[1],values[2],values[3],values[4]])
    index=0
    def NegativeExp(x,a,b,c,d,f,g,h):
        return a*np.exp(-b*x)+c*np.exp(-d*x)+f*np.exp(-g*x)+h
    
    for indexForLoop1 in range(resolution):
        amountToUse=yaxis[indexForLoop1]
        for indexForLoop2 in range(resolution):
            #print("__________________________")
            
            mysteryToUse=xaxis[indexForLoop2]
            for indexForLoop3 in range(resolution):
                print(index+1,"out of ", len(xaxis)**3)
                effToUse=zaxis[indexForLoop3]
                guess=[values[0],amountToUse,effToUse,values[3],mysteryToUse]
                errorResults=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(21.4603,21.4603),(amountToUse,amountToUse),(effToUse,effToUse),(0.,1000.),(mysteryToUse,mysteryToUse)])
                errorValues=errorResults['x']
                currentL=FinderYear([errorValues[0],amountToUse,effToUse,errorValues[3],mysteryToUse])
                print(np.abs(currentL-minimumL),"Diff")
                if np.abs(currentL-minimumL)<=0.5:
                    SDAmount.append(amountToUse)
                    SDEff.append(effToUse)
                    SDMystery.append(mysteryToUse)
                    print("New point within 1sigma at ",amountToUse,effToUse,mysteryToUse)
                errorAmountMysteryEff.append(currentL)
                index=index+1
    Z1=np.array(errorAmountMysteryEff)
    Z2=Z1.reshape(len(xaxis),len(yaxis),len(zaxis))
    print(Z2,"Z2 in fit 3")
    fig,ax=plt.subplots(1,1)
    X1,X2,X3=np.meshgrid(xaxis,yaxis,zaxis)
    levels=np.linspace(minimumL,minimumL+2,10)
    #cp=ax.contourf(X1,X2,X3,levels=levels)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    img=ax.scatter(X1,X2,X3,c=(errorAmountMysteryEff),cmap=plt.hot())
    fig.colorbar(img)
    plt.title("Error Landscape")
    #cp2=ax.contour(X1,X2,Z2,[FinderYear([values[0],values[1],values[2],values[3],values[4]])+0.5],colors='red')
# =============================================================================
#     for item in cp2.collections:
#         for i in item.get_paths():
#             v=i.vertices
#             vx=v[:,0]
#             vy=v[:,1]
#             print(vx,vy,"allegedly xy points on the l=min+0.5 line")
#             print("From", min(vx)," to ",max(vx)," in the mystery axis:",values[4]," (+/-) ", np.abs(max(vx)-min(vx))/2)
#             print("From", min(vy)," to ",max(vy)," in the efficiency axis:",values[1]," (+/-) ", np.abs(max(vy)-min(vy))/2)
# =============================================================================
    plt.xlabel("Neutron Rate from Mystery Isotope")
    plt.ylabel("Amount")
    plt.show()
    fig1,ax1=plt.subplots()
    for i in range(len(SDAmount)):
        neutronRange=CfCmDataGeneratorT0.DataGenerator(rangeOfValues,SDAmount[i],SDEff[i],1/(1+values[2]),1-(1/(1+values[2])),SDMystery[i],resolution*[14402],resolution*[0.02],True,True)
        print(neutronRange,"neutornRange")
        #ax1.set_xticklabels([0,2015,2017,2019,2021,2023,2025,2027,2029])

        plt.xlabel("Time since February 1995 (Years)")
        plt.ylabel("Fissions/Second")
        plt.title("Fission Rate for the next 10 years")
        valueArray=np.array(rangeOfValues)
        neutronRangeArray=np.array(neutronRange)
        ax1.scatter(valueArray,neutronRangeArray,s=2)
        preShapedNeutrons.append((neutronRange))
        #preShapedNeutrons=np.append(preShapedNeutrons,[neutronRange])
                
# =============================================================================
#             plt.show()
#             img=ax.scatter(X1,X2,X3,c=(testListInFirstLoopL),cmap=plt.hot())
#             fig.colorbar(img)
# =============================================================================
            #neutronMax=np.reshape(preShapedNeutrons,(-1,resolution))
    print(preShapedNeutrons,"preshapedneutrons")
    preShapedNeutrons=np.swapaxes(preShapedNeutrons,0,1)
    print(preShapedNeutrons,"post shaped")
    for i in range(resolution):
        neutronMax.append(max(preShapedNeutrons[i]))
        neutronMin.append(min(preShapedNeutrons[i]))
    print(neutronMax,"neutron max")
    print(neutronMin,"neutron min")
    neutronUncertainty=[np.abs(n2-n1) for n1,n2 in zip(neutronMax,neutronMin)]
        
    print(neutronUncertainty,"Uncertainty")
        

   # p0=[10,1,5,1,1,1,1]
    bestFitNeutrons=CfCmDataGeneratorT0.DataGenerator(rangeOfValues,values[1],values[2],1/(1+values[3]),1-(1/(1+values[3])),values[4],resolution*[14402],resolution*[0.0671],False,True)
    print(bestFitNeutrons,"bestfitneutrons")
    #print(neutronRange2,"neutron range")
        #print(len(neutronRange2),"len neutron range")
    #popt,pcov=curve_fit(NegativeExp,rangeOfValues,neutronRange2,p0,maxfev=800000)
    analyticInterp=[]
    #for i in range(len(rangeOfValues)):
   #     analyticInterp.append(bestFitNeutrons)
        #analyticInterp.append(popt[0]*np.exp(-popt[1]*rangeOfValues[i])+popt[2]*np.exp(-popt[3]*rangeOfValues[i])+popt[4]*np.exp(-popt[5]*rangeOfValues[i])+popt[6])
    print(analyticInterp)
    neutronUncertaintyPercentageTop=[n2/n1 for n1,n2 in zip(neutronMax,bestFitNeutrons)]
    neutronUncertaintyPercentageBottom=[n2/n1 for n1,n2 in zip(neutronMin,bestFitNeutrons)]
    #plt.plot(rangeOfValues,analyticInterp,color='r',linewidth=1)
    #plt.scatter([27.3255],[8.288102],color='gold')

        
        #print(neutronMin,"Min range of neutrons")
        #print(neutronMax,"Max range of neutrons")
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    levels=np.linspace(minimumL,minimumL+60,10)
        #cp=ax.contourf(X1,X2,levels=levels)
    print("after cp")
    #1 fig.colorbar(cp)
    print("after fig colorbar")
    plt.title("Fissions Over Time")
            #cp2 = ax.contour(X1,X2, Z2, [FinderYear([values[0],values[1],values[2],values[3]])+0.5])
            #for item in cp2.collections:
             #   for i in item.get_paths():
              #      v=i.vertices
               #     vx=v[:,0]
                #    vy=v[:,1]
                 #   print(vx,vy,"Allegedly points on the L=min+0.5 line")
                  #  print("From", min(vx)," to ",max(vx)," in the proportion axis"," (+/-) ", np.abs(max(vx)-min(vx))/2)
                   # print("From", min(vy)," to ",max(vy)," in the efficiency axis"," (+/-) ", np.abs(max(vy)-min(vy))/2)
            #plt.scatter(scatterTestX,scatterTestY,color='r')
    print("After cp2")
    plt.show()
    fig2,ax2=plt.subplots()
    ax2.plot(rangeOfValues,neutronUncertainty,label="Uncertainty in Neutron rate over time")
    plt.legend()
    plt.xlabel("Age of Sample (Years)")
    plt.ylabel("Fission Rate (Fissions/s)")
    plt.title("Uncertainty in Fission Rate over Time")
    plt.show()
    
    fig3,ax3=plt.subplots()
    ax3.plot(rangeOfValues,neutronUncertaintyPercentageTop,label="Max")
    ax3.plot(rangeOfValues,neutronUncertaintyPercentageBottom,label="Min")
    plt.legend()
    plt.xlabel("Age of Sample (Years)")
    plt.ylabel("Proportional Error")
    plt.title("Proportional Error over Time")
    plt.show()
        
    print("Error Analysis of the Amount and Mystery Isotope took", np.abs(ErrorTime-time.time())," seconds to run.")
    
    

def ErrorComputer4D(values):

    print("Initializing 4D Error Analysis....")
    ErrorTime=time.time()
    resolution=10
    errorAmountMysteryEff=[]
    preShapedNeutrons=[]
    rangeOfValues=np.linspace(21,36,resolution)
    SDAmount,SDEff,SDProp,SDMystery=[],[],[],[]
    neutronMin,neutronMax=[],[]
    waxis=np.linspace(values[3]-0.01,values[3]+0.01,resolution)
    xaxis=np.linspace(values[4]-0.03,values[4]+0.03,resolution)
    yaxis=np.linspace(values[1]-12000,values[1]+12000,resolution)
    zaxis=np.linspace(values[2]-0.0002,values[2]+0.0002,resolution)
    minimumL=FinderYear([values[0],values[1],values[2],values[3],values[4]])
    index=0
    def NegativeExp(x,a,b,c,d,f,g,h):
        return a*np.exp(-b*x)+c*np.exp(-d*x)+f*np.exp(-g*x)+h
    
    for indexForLoop1 in range(resolution):
        amountToUse=yaxis[indexForLoop1]
        for indexForLoop2 in range(resolution):
            #print("__________________________")
            
            mysteryToUse=xaxis[indexForLoop2]
            for indexForLoop3 in range(resolution):
                effToUse=zaxis[indexForLoop3]
                for indexForLoop4 in range(resolution):
                    propToUse=waxis[indexForLoop4]
                    print(index+1,"out of ", len(xaxis)**4)
                    
                    guess=[values[0],amountToUse,effToUse,propToUse,mysteryToUse]
                    errorResults=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(21.4603,21.4603),(amountToUse,amountToUse),(effToUse,effToUse),(propToUse,propToUse),(mysteryToUse,mysteryToUse)])
                    errorValues=errorResults['x']
                    currentL=FinderYear([errorValues[0],amountToUse,effToUse,propToUse,mysteryToUse])
                    print(np.abs(currentL-minimumL),"Diff")
                    if np.abs(currentL-minimumL)<=0.5:
                        SDAmount.append(amountToUse)
                        SDEff.append(effToUse)
                        SDMystery.append(mysteryToUse)
                        SDProp.append(propToUse)
                        print("New point within 1sigma at ",amountToUse,effToUse,propToUse,mysteryToUse)
                    errorAmountMysteryEff.append(currentL)
                    index=index+1
   # Z1=np.array(errorAmountMysteryEff)
   # Z2=Z1.reshape(len(xaxis),len(yaxis),len(zaxis))
   # print(Z2,"Z2 in fit 3")
    #fig,ax=plt.subplots(1,1)
    #X1,X2,X3=np.meshgrid(xaxis,yaxis,zaxis)
    #levels=np.linspace(minimumL,minimumL+2,10)
    #cp=ax.contourf(X1,X2,X3,levels=levels)
   # fig=plt.figure()
    #ax=fig.add_subplot(111,projection='3d')
    #img=ax.scatter(X1,X2,X3,c=(errorAmountMysteryEff),cmap=plt.hot())
    #fig.colorbar(img)
    #plt.title("Error Landscape")
    #cp2=ax.contour(X1,X2,Z2,[FinderYear([values[0],values[1],values[2],values[3],values[4]])+0.5],colors='red')
# =============================================================================
#     for item in cp2.collections:
#         for i in item.get_paths():
#             v=i.vertices
#             vx=v[:,0]
#             vy=v[:,1]
#             print(vx,vy,"allegedly xy points on the l=min+0.5 line")
#             print("From", min(vx)," to ",max(vx)," in the mystery axis:",values[4]," (+/-) ", np.abs(max(vx)-min(vx))/2)
#             print("From", min(vy)," to ",max(vy)," in the efficiency axis:",values[1]," (+/-) ", np.abs(max(vy)-min(vy))/2)
# =============================================================================
    plt.xlabel("Neutron Rate from Mystery Isotope")
    plt.ylabel("Amount")
    plt.show()
    fig1,ax1=plt.subplots()
    for i in range(len(SDAmount)):
        print("initializing loop")
        neutronRange=CfCmDataGeneratorT0.DataGenerator(rangeOfValues,SDAmount[i],SDEff[i],1-SDProp[i],SDProp[i],SDMystery[i],resolution*[14402],resolution*[0.02],True,True)
        print(neutronRange,"neutornRange")
        plt.xlabel("Time (Yeras)")
        plt.ylabel("Fissions/Second")
        plt.title("Fission Rate for the next 10 years")
        valueArray=np.array(rangeOfValues)
        neutronRangeArray=np.array(neutronRange)
        ax1.scatter(valueArray,neutronRangeArray,s=2)
        preShapedNeutrons.append((neutronRange))
        #preShapedNeutrons=np.append(preShapedNeutrons,[neutronRange])
                
# =============================================================================
#             plt.show()
#             img=ax.scatter(X1,X2,X3,c=(testListInFirstLoopL),cmap=plt.hot())
#             fig.colorbar(img)
# =============================================================================
            #neutronMax=np.reshape(preShapedNeutrons,(-1,resolution))
    print(preShapedNeutrons,"preshapedneutrons")
    preShapedNeutrons=np.swapaxes(preShapedNeutrons,0,1)
    print(preShapedNeutrons,"post shaped")
    for i in range(resolution):
        neutronMax.append(max(preShapedNeutrons[i]))
        neutronMin.append(min(preShapedNeutrons[i]))
    print(neutronMax,"neutron max")
    print(neutronMin,"neutron min")
    neutronUncertainty=[np.abs(n2-n1) for n1,n2 in zip(neutronMax,neutronMin)]
        
    print(neutronUncertainty,"Uncertainty")
        

   # p0=[10,1,5,1,1,1,1]
    bestFitNeutrons=CfCmDataGeneratorT0.DataGenerator(rangeOfValues,values[1],values[2],1/(1+values[3]),1-(1/(1+values[3])),values[4],resolution*[14402],resolution*[0.0671],False,True)
    print(bestFitNeutrons,"bestfitneutrons")
    #print(neutronRange2,"neutron range")
        #print(len(neutronRange2),"len neutron range")
    #popt,pcov=curve_fit(NegativeExp,rangeOfValues,neutronRange2,p0,maxfev=800000)
    analyticInterp=[]
    #for i in range(len(rangeOfValues)):
   #     analyticInterp.append(bestFitNeutrons)
        #analyticInterp.append(popt[0]*np.exp(-popt[1]*rangeOfValues[i])+popt[2]*np.exp(-popt[3]*rangeOfValues[i])+popt[4]*np.exp(-popt[5]*rangeOfValues[i])+popt[6])
    print(analyticInterp)
    neutronUncertaintyPercentageTop=[n2/n1 for n1,n2 in zip(neutronMax,bestFitNeutrons)]
    neutronUncertaintyPercentageBottom=[n2/n1 for n1,n2 in zip(neutronMin,bestFitNeutrons)]
    #plt.plot(rangeOfValues,analyticInterp,color='r',linewidth=1)
    #plt.scatter([27.3255],[8.288102],color='gold')

        
        #print(neutronMin,"Min range of neutrons")
        #print(neutronMax,"Max range of neutrons")
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    levels=np.linspace(minimumL,minimumL+60,10)
        #cp=ax.contourf(X1,X2,levels=levels)
    print("after cp")
    #1 fig.colorbar(cp)
    print("after fig colorbar")
    plt.title("Fissions Over Time")
            #cp2 = ax.contour(X1,X2, Z2, [FinderYear([values[0],values[1],values[2],values[3]])+0.5])
            #for item in cp2.collections:
             #   for i in item.get_paths():
              #      v=i.vertices
               #     vx=v[:,0]
                #    vy=v[:,1]
                 #   print(vx,vy,"Allegedly points on the L=min+0.5 line")
                  #  print("From", min(vx)," to ",max(vx)," in the proportion axis"," (+/-) ", np.abs(max(vx)-min(vx))/2)
                   # print("From", min(vy)," to ",max(vy)," in the efficiency axis"," (+/-) ", np.abs(max(vy)-min(vy))/2)
            #plt.scatter(scatterTestX,scatterTestY,color='r')
    print("After cp2")
    plt.show()
    fig2,ax2=plt.subplots()
    ax2.plot(rangeOfValues,neutronUncertainty,label="Uncertainty in Neutron rate over time")
    plt.legend()
    plt.xlabel("Age of Sample (Years)")
    plt.ylabel("Fission Rate (Fissions/s)")
    plt.title("Uncertainty in Fission Rate over Time")
    plt.show()
    
    fig3,ax3=plt.subplots()
    ax3.plot(rangeOfValues,neutronUncertaintyPercentageTop,label="Max")
    ax3.plot(rangeOfValues,neutronUncertaintyPercentageBottom,label="Min")
    plt.legend()
    plt.xlabel("Age of Sample (Years)")
    plt.ylabel("Proportional Error")
    plt.title("Proportional Error over Time")
    plt.show()
        
    print("Error Analysis of the Amount and Mystery Isotope took", np.abs(ErrorTime-time.time())," seconds to run.")
def ErrorComputer3DPrototype(values):
    testListInFirstLoopL=[]
    testListInFirstLoopEff=[]
    testListInFirstLoopProp=[]
    testListInFirstLoopAmount=[]
    listAnalyticPropEff=[]
    AmountInSD=[]
    EffInSD=[]
    MysteryInSD=[]
    indexForEffLoop=0
    indexForEffLoop2=0
    paramResolution=4
    index=0
    neutronMax=[]
    neutronMin=[]
    neutronUncertainty=[]
    neutronUncertaintyPercentage=[]
    preShapedNeutrons=[]
    scatterTestX=[]
    scatterTestY=[]
    ijk=0
    keepGoingDate=True
    minimumL=FinderYear([values[0],values[1],values[2],values[3],values[4]])
    print(minimumL,"minL")
    xgraph=np.linspace(values[1]-10000,values[1]+10000,paramResolution)
    ygraph=np.linspace(values[2]-0.001,values[2]+0.001,paramResolution)
    zgraph=np.linspace(values[4]-0.2,values[4]+0.2,paramResolution)
    for indexForEffLoop in range(paramResolution):
        #dateToUse=aroundMinDate+(indexForDateLoop*0.16)-(dateAmountSteps/2)*0.16
        effToUse=ygraph[indexForEffLoop]
        testListInFirstLoopEff.append(effToUse)
        print(testListInFirstLoopEff,"testlist")


        for indexForEffLoop2 in range(paramResolution):
            amountToUse=xgraph[indexForEffLoop2]
            testListInFirstLoopProp.append(amountToUse)
            if amountToUse in testListInFirstLoopProp:
                print("Duplicate Detected in Prop! Hopefully omitting!")
            else:
                testListInFirstLoopProp.append(amountToUse)
                
            for indexForEffLoop3 in range(paramResolution):
                mysteryToUse=zgraph[indexForEffLoop3]
                print("__________________________________")
                print(ijk,"out of", paramResolution**3)

            
            #amountToUse=aroundMinAmount+(indexForDateLoop2*18000000)-(amountDateSteps/2)*18000000
            

            # currentL=FinderYear([dateToUse,amountToUse,values[2],values[3]])
                guess=[values[0],amountToUse,effToUse,mysteryToUse,values[4]]
                #tempResults=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(20.5589,20.5589),(amountToUse,amountToUse),(effToUse,effToUse),(propToUse,propToUse)])
                tempResults=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(21.2301,21.2301),(amountToUse,amountToUse),(effToUse,effToUse),(values[3],values[3]),(mysteryToUse,mysteryToUse)])
                #tempResults=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(15,50),(100,5353300000),(effToUse,effToUse),(propToUse,propToUse)])

                #print(tempResults,"tempresults")
                tempValues=tempResults['x']
                #print(tempValues,"temp values")
                currentL=FinderYear([tempValues[0],amountToUse,effToUse,values[3],mysteryToUse])
                if np.abs(currentL-minimumL)>=0.5:
                    AmountInSD.append(amountToUse)
                    EffInSD.append(effToUse)
                    MysteryInSD.append(mysteryToUse)
                #print(currentL,"currentL")
                testListInFirstLoopL.append(currentL)
                #testListInFirstLoopAmount.append(amountToUse)

# =============================================================================

                ijk=ijk+1
                if amountToUse in testListInFirstLoopAmount:
                    print("Duplicate Detected in Amount, Hopefully Omitting!")
                else:
                    testListInFirstLoopAmount.append(amountToUse)

        

    


    #testGridAge,testGridAmount=np.meshgrid(listAnalyticDate,listAnalyticAmountDate)
    # print(testGridAge,"meshgrid1")
    #print(testGridAmount,"meshgrid2")
    # =============================================================================
    Z1=np.array(testListInFirstLoopL)  
    #print(Z1,"Z1")
    Z2=Z1.reshape(len(testListInFirstLoopEff),len(xgraph),len(testListInFirstLoopAmount))
    #print(Z2,"Z2")
    fig,ax=plt.subplots(1,1)
    #print("after fig")
    X1,X2,X3=np.meshgrid(xgraph,ygraph,zgraph)
    #print(X1,"x1")
    #print(X2,"x2")
    #print(X3,"X3")
    #print(Z2,"z2")
    fig=plt.figure()
    #ax=fig.add_subplot(111,projection='3d')
    #print(xgraph,"xgraph")
    #print(ygraph,"ygraph")
    #print(zgraph,"zgraph")
    print(AmountInSD,"Amount in SD")
    print(len(AmountInSD),"Amount in SD Length")
    print(EffInSD,"Eff in SD")
    print(len(EffInSD),"Efficiency in SD Length")

    resolution=100
    rangeOfValues=np.linspace(20,50,resolution)   
    def NegativeExp(x,a,b,c,d,f,g,h):
        return a*np.exp(-b*x)+c*np.exp(-d*x)+f*np.exp(-g*x)+h
    
    for i in range(len(AmountInSD)):    
        print(i,"out of ",len(AmountInSD))
        
        
            #DataGenerator([33.932,34.06,34.178,34.255,39.4137,39.6,40],932560648,0.486700153,0.856340985,0.143659015,[14402,14402,14402,14402,25786,25786,25786],[0.0671,0.0671,0.0671,0.0671,0.0671,0.0671,0.0671],True,True)
            #Change rate - it is detected not emitted
            
        neutronRange=(CfCmDataGeneratorT0.DataGenerator(rangeOfValues,AmountInSD[i],EffInSD[i],1/(1+values[3]),1-(1/(1+values[3])),MysteryInSD[i],resolution*[14402],resolution*[0.0671],True,True))
            
            

            #print(pcov)
            #list_dict = {'Years':rangeOfValues, 'Neutron Rate':neutronRange}
            #df = pd.DataFrame(list_dict) 
            #df.to_csv('listOfValues.csv', index=False) 
            #print(popt,"values")
            #analyticInterp=[]
            #for i in range(len(rangeOfValues)):
           #     analyticInterp.append(popt[0]*np.exp(-popt[1]*rangeOfValues[i])+popt[2]*np.exp(-popt[3]*rangeOfValues[i])+popt[4]*np.exp(-popt[5]*rangeOfValues[i])+popt[6])
            #plt.plot(rangeOfValues,analyticInterp,color='r')
        plt.xlabel("Time (Years)")
        plt.ylabel("Fissions/Second")
        plt.title("Fission Rate for the next 10 Years")
            #print(len(rangeOfValues),"range of values")
           # print(len(neutronRange),"neutronrange")
            #print(rangeOfValues)
            #print(neutronRange)
        valueArray=np.asarray(rangeOfValues)
        neutronRangeArray=np.asarray(neutronRange)
            
            #print(valueArray,"Valuearray")
            #print(neutronRangeArray,"Neutron Range Array")
        plt.scatter(valueArray,neutronRangeArray,s=1)
            #plt.plot()
            #plt.scatter([20.5589,20.6869,20.8049,20.8819,26.0406],[16.415289,16.101305,15.844049,15.654701,8.288102],color='gold')
        #plt.show()
        preShapedNeutrons.append((neutronRange))
        
# =============================================================================
#             plt.show()
#             img=ax.scatter(X1,X2,X3,c=(testListInFirstLoopL),cmap=plt.hot())
#             fig.colorbar(img)
# =============================================================================
    #neutronMax=np.reshape(preShapedNeutrons,(-1,resolution))
    preShapedNeutrons=np.swapaxes(preShapedNeutrons,0,1)
    print(preShapedNeutrons[0],"pre shaped")
    for i in range(resolution):
        neutronMax.append(max(preShapedNeutrons[i]))
        neutronMin.append(min(preShapedNeutrons[i]))
    print(neutronMax,"neutron max")
    print(neutronMin,"neutron min")
    neutronUncertainty=[np.abs(n2-n1) for n1,n2 in zip(neutronMax,neutronMin)]
    
    print(neutronUncertainty,"Uncertainty")
    
    p0=[10,1,5,1,1,1,1]
    neutronRange2=CfCmDataGeneratorT0.DataGenerator(rangeOfValues,values[1],values[2],1/(1+values[3]),1-(1/(1+values[3])),values[4],resolution*[14402],resolution*[0.0671],False,True)
    #print(neutronRange2,"neutron range")
    #print(len(neutronRange2),"len neutron range")
    popt,pcov=curve_fit(NegativeExp,rangeOfValues,neutronRange2,p0,maxfev=800000)
    analyticInterp=[]
    for i in range(len(rangeOfValues)):
        analyticInterp.append(popt[0]*np.exp(-popt[1]*rangeOfValues[i])+popt[2]*np.exp(-popt[3]*rangeOfValues[i])+popt[4]*np.exp(-popt[5]*rangeOfValues[i])+popt[6])
    
    neutronUncertaintyPercentageTop=[n2/n1 for n1,n2 in zip(neutronMax,analyticInterp)]
    neutronUncertaintyPercentageBottom=[n2/n1 for n1,n2 in zip(neutronMin,analyticInterp)]
    plt.plot(rangeOfValues,analyticInterp,color='r',linewidth=1)
    plt.scatter([27.3255],[8.288102],color='gold')
    
    #print(neutronMin,"Min range of neutrons")
    #print(neutronMax,"Max range of neutrons")

    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    levels=np.linspace(minimumL,minimumL+60,10)
    print("after cp")
    print("after fig colorbar")
    plt.title("Fissions Over Time")
    print("After cp2")
    plt.show()
    fig2,ax2=plt.subplots()
    ax2.plot(rangeOfValues,neutronUncertainty)
    plt.xlabel("Age of Sample (Years)")
    plt.ylabel("Fission Rate (Fissions/s)")
    plt.title("Uncertainty in Fission Rate over Time")
    plt.show()
    
    fig3,ax3=plt.subplots()
    ax3.plot(rangeOfValues,neutronUncertaintyPercentageTop)
    ax3.plot(rangeOfValues,neutronUncertaintyPercentageBottom)
    plt.xlabel("Age of Sample (Years)")
    plt.ylabel("Proportional Error")
    plt.title("Proportional Error over Time")

    # =============================================================================







print("after plt show (end before time)")
    

#This funciton is called by other programs that wish to call the likelihood function. Data is a list of csvs, guess is the x0 for the fitter, and runLengths is a list of the length of the run of each csv data file in seconds
def DateFitter(Data,guess,RunLengths,BackgroundRate):
    time_in_fitter=time.time()
    global runLengths
    global backgroundRate
    global data
    runLengths=RunLengths
    backgroundRate=BackgroundRate
    data=Data
    print("Initializing Likelihood Fit....")
    results=scipy.optimize.minimize(FinderYear,guess,method='nelder-mead',bounds=[(21.4603,21.4603),(1000000,1000000000),(0.,1.),(0.,1000.),(0,100.)],options={'maxfev':100000,'disp':True})
    values=results['x']
    print(values[0],"The Age of the Sample at the first SNOLAB run considered. This value is not explicitely fit since the precise age is treated as known")
    print(values[1],"The Amount of Fissions expected at t0 in a",runLengths[0],"amount of seconds")
    print(values[2],"The Efficiency of the HALO detector in the position where these runs were taken")
    print(values[3],"The Mass Ratio between Cf252/Cf250 at t0")
    print(values[4],"The Background rate of Neutrons from a long lived mystery isotope")
    #The Minimum Log Likelihood value is computed, this is useful for later in the computation of errors
    minimumL=FinderYear([values[0],values[1],values[2],values[3],values[4]])
    print(np.abs(time_in_fitter-time.time()),"Time in fitter")
    Error3d=ErrorComputer3D(values)
    return values
    
#Print the amount of time in seconds the program took to run
print("My program took", time.time()-start_time,"seconds to run")