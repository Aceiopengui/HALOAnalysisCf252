import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import chisquare
from scipy.optimize import differential_evolution
import numdifftools
import time

start_time=time.time()

decayConst252=np.log(2)/2.65
decayConst250=np.log(2)/13.08
decayConst248=np.log(2)/348000
decayConstCo60=np.log(2)/5.27
mult252=3.735
mult250=3.50
mult248=3.13
chanAlpha252,chanAlpha250,chanAlpha248=0.96908,0.99923,0.9161
chanFiss252,chanFiss250,chanFiss248=1-chanAlpha252,1-chanAlpha250,1-chanAlpha248
ygraph=[]
    
ygraphcf252=[]
ygraphcf250=[]
ygraphcm248init0=[]
ygraphearlycm248=[]
withFirst=True
fittedEff=False
newFit=True
withColinRun=False
oldEff=True
tightWindow=False
#If we see the neutron rate given here, then that is evidence the strength reported is wrong
directMeasurement=False
if withFirst:
    if tightWindow:
        ages=[1.2849,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326,27.8438]
        neutronRates=[8965,215.88,70.48,66.30,64.83,63.77,63.09,31.98,30.31]
        errors2=[89.65,2.16,0.7,0.66,0.65,0.64,0.63,0.32,0.3]
    else:
        print("2")
        ages=[1.2849,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326,27.8438]
        neutronRates=[8965,215.88,75.305,70.861,69.263,68.164,67.4384,34.304,32.475]
        errors2=[89.65,2.16,0.75,0.71,0.69,0.68,0.76,0.34,0.32]
if withFirst==False:
    if directMeasurement:
        ages=[21.8438,21.9726,22.0932,22.1699,27.326,27.8438]
        neutronRates=[34.013,33.246,32.719,32.370,16.466,15.488]
        errors2=[0.4,0.37,0.35,0.34,0.34,0.34,0.16,0.15]
    else:
        print("4")
        ages=[21.4603,21.8438,21.9726,22.0932,22.1699,27.326,27.8438]
        neutronRates=[75.305,70.861,69.263,68.164,67.4384,34.304,32.475]
        errors2=[0.75,0.71,0.69,0.68,0.76,0.34,0.32]

#if directMeasurement==False:
# =============================================================================
#     if newFit==True:
#         if withColinRun:
#             if withFirst:
#                 ages=np.array([1.2849,16.063,21.2274,21.4603,21.8438,21.9726,22.0932,22.1699,27.326])
#                 neutronRates=np.array([8965,215.88,74.63,72.4,68.12,66.59,65.53,64.83,32.98])
#                 errors2=np.array([89.65,2.16,0.8,0.75,0.69,0.67,0.66,0.66,0.33]) 
#     
#     
#     
#             if withFirst==False:
#                 ages=np.array([21.2274,21.4603,21.8438,21.9726,22.0932,22.1699,27.326])
#                 neutronRates=np.array([74.63,72.4,68.12,66.59,65.53,64.83,32.98])
#                 errors2=np.array([0.8,0.75,0.69,0.67,0.66,0.66,0.33]) 
#                 
#         else:
#             if oldEff:
#                 if withFirst:
#                     ages=np.array([1.2849,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326])
#                     neutronRates=np.array([8965,215.88,72.4,68.12,66.59,65.53,64.83,32.98])
#                     errors2=np.array([89.65,2.16,0.75,0.69,0.67,0.66,0.66,0.33])  
#                 if withFirst==False:
#                     ages=np.array([21.4603,21.8438,21.9726,22.0932,22.1699,27.326])
#                     neutronRates=np.array([72.4,68.12,66.59,65.53,64.83,32.98])
#                     errors2=np.array([0.75,0.69,0.67,0.66,0.66,0.33]) 
#             if oldEff==False:
#                                 #Efficiency is 48%
#                 ages=np.array([1.2849,16.063,21.8438,21.9726,22.0932,22.1699,27.326,27.8438])
#                 neutronRates=np.array([8965,215.88,70.86,69.263,68.164,67.438,34.30])
#                 errors2=np.array([89.65,2.16,0.69,0.67,0.66,0.66,0.33])
#             
#             
#                 
#     if fittedEff==True:
#         if withFirst==True:
#             ages=np.array([1.2849,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326])
#             neutronRates=np.array([8965,215.88,74.77,70.36,68.77,67.38,66.96,34.06])
#             errors2=np.array([89.65,2.16,0.35,0.3,0.3,0.29,0.29,0.15])
#         
#         if withFirst==False:
#             ages=np.array([21.4603,21.8438,21.9726,22.0932,22.1699,27.326])
#             neutronRates=np.array([74.73,70.86,69.26,68.16,67.44,34.30])
#             errors2=np.array([3.15,2.95,2.89,2.84,2.81,1.43])
#             
#     if fittedEff==False and newFit==False:
#     
#         if withFirst==True:
#             ages=np.array([1.2849,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326])
#             neutronRates=np.array([8965,215.88,74.73,70.86,69.26,68.16,67.44,34.30])
#             errors2=np.array([269,6.48,3.15,2.95,2.89,2.84,2.81,1.43])
#         
#         if withFirst==False:
#             ages=np.array([21.4603,21.8438,21.9726,22.0932,22.1699,27.326])
#             neutronRates=np.array([74.73,70.86,69.26,68.16,67.44,34.30])
#             errors2=np.array([3.15,2.95,2.89,2.84,2.81,1.43])
# =============================================================================
        



Version=6
#Version 0 fixes mass proportion between Cf isotopes, floats Curium
#Version 1 floats everything that can be floated (Floats Everything) (Least useful) 
#Version 2 assumes cm248(0)=0, floats 252/250 proportion
#Version 3 assumes cm248(0)=0 and fixed 252/250 Proportion (Fixes everything)
#Version 4 includes mystery isotope and fixes 252/250 Proportion (Assumes cm248(0)=0)
#Version 5 includes Cobalt-60 as a fissile isotope
#Version 6 is a plot that demonstrates the runs taken at SNOLAB all have effectively one half life
#Version 7 fixes the isotopic ratios and cm248(0)=0 but adds an isotope with floating half life and amplitude
if Version==0:
    print("Version 0 activating")
    def Curve(guess):
        #x=[1.3589,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326]
        #neutronRates=np.array([8965,215.88,74.73,70.86,69.26,68.16,67.44,34.30])
        a=guess[0]
        #f=guess[1]
        f=(a/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)
        b=guess[1]
    
        chisq=0

    
        for i in range(len(ages)):
            #func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+(chanAlpha252*((a*f/(chanFiss252*decayConst252*mult252))-(a*f/(chanFiss252*decayConst252*mult252))*np.exp(-decayConst252*ages[i]))*np.exp(-decayConst248*ages[i])))+b*np.exp(-decayConst248*ages[i])
            #########func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+((((a*f)/(chanFiss252*decayConst252*mult252))*decayConst248*chanAlpha252)*(ages[i]+(1/decayConst252)*np.exp(-decayConst252*ages[i]))+b)*(decayConst248*chanFiss248*mult248))
            #WORKING#func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((decayConst248*chanAlpha252*a*ages[i])+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*ages[i]))+b)*np.exp(-decayConst248*ages[i])
            func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*ages[i]))*(decayConst248*mult248*chanFiss248)+b)*np.exp(-decayConst248*ages[i])
            chisq+=((neutronRates[i]-func)**2)/errors2[i]
        #print(chisq,"chisq")
        #print(guess)
        return chisq






    xgraph=np.linspace(0,30,200)
    guess=[10000,10]
    mintest=minimize(Curve,guess,method='nelder-mead',options={'maxfev':9000000,'disp':True})
    #mintest=differential_evolution(Curve,bounds=[(0,10000000),(0.,1.),(0,10000000)],x0=guess)
    values=mintest['x']
    
    
    
    
    Hfun = numdifftools.Hessian(Curve, full_output="true")
    hessian_ndt, info = Hfun(values)
    
    se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
    print("Fit Uncertainty: " + str(se))
    

    print(values,"avlues")
    
    a=values[0]
    
    #f=values[1]
    f=(a/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)
    print(f,"f")
    b=values[1]
    cf252mass=a/(decayConst252*mult252*chanFiss252)
    cf250mass=f/(decayConst250*mult250*chanFiss250)
    cm248mass=b/(decayConst248*mult248*chanFiss248)
    print(cf252mass,cf250mass,cm248mass,"mass")
    
    print((se[0]/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250))
    
    cf252masserror=se[0]/(decayConst252*mult252*chanFiss252)
    cf250masserror=se[0]/(decayConst252*mult252*chanFiss252*8.164)
    cm248masserror=se[1]/(decayConst248*mult248*chanFiss248)
    print(cf252masserror,cf250masserror,cm248masserror,"masserror")
    
    
    print(100*cf252mass/(cf252mass+cf250mass+cm248mass),"mass percentage of cf252")
    print(100*cf250mass/(cf252mass+cf250mass+cm248mass),"mass percentage of cf250")
    print(100*cm248mass/(cf252mass+cf250mass+cm248mass),"mass percentage of cm248")

    cf252propmasserror=np.sqrt(((cf250mass+cm248mass)/((cf252mass+cf250mass+cm248mass)**2)*cf252masserror)**2+((-(cf252mass/(cf252mass+cf250mass+cm248mass)**2))*cf250masserror)**2+((-(cf252mass/(cf252mass+cf250mass+cm248mass)**2))*cm248masserror)**2)
    cf250propmasserror=np.sqrt(((cf252mass+cm248mass)/((cf250mass+cf252mass+cm248mass)**2)*cf250masserror)**2+((-(cf250mass/(cf250mass+cf252mass+cm248mass)**2))*cf252masserror)**2+((-(cf250mass/(cf250mass+cf252mass+cm248mass)**2))*cm248masserror)**2)
    
    
    cm248propmasserror=np.sqrt(((cf250mass+cf252mass)/((cf252mass+cf250mass+cm248mass)**2)*cm248masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf250masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf252masserror)**2)
    
    print(100*cf252propmasserror,"big equation252")
    print(100*cf250propmasserror,"big equation250")
    print(100*cm248propmasserror,"big equation248")
    
    normedprop=(values[0]/(decayConst252*chanFiss252))/(values[0]/(decayConst252*chanFiss252)+(1-values[0])/(decayConst250*chanFiss250))
    
    
    
    
#WORKING#ygraph=(a*np.exp(-decayConst252*xgraph))+(f*np.exp(-decayConst250*xgraph))+((decayConst248*chanAlpha252*a*xgraph)+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*xgraph))+b)*np.exp(-decayConst248*xgraph)
    
    ygraph=(a*np.exp(-decayConst252*xgraph))+(f*np.exp(-decayConst250*xgraph))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)+b)*np.exp(-decayConst248*xgraph)
    
    ygraphcf252=a*np.exp(-decayConst252*xgraph)
    ygraphcf250=f*np.exp(-decayConst250*xgraph)
    ygraphcm248init0=(chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)
    ygraphearlycm248=b*np.exp(-decayConst248*xgraph)
    
    fig,ax=plt.subplots(dpi=250)
    #ax.set_xlim(20,28)
    #ax.set_ylim(-1,200)
    ax.plot(xgraph,ygraph,label='Total Neutron Rate')
    ax.plot(xgraph,ygraphcf252,label='Neutron Rate as a Result of Cf252')
    ax.plot(xgraph,ygraphcf250,label='Neutron Rate as a Result of Cf250')
    ax.plot(xgraph,ygraphcm248init0,label='Neutron Rate as a result of Cm248 being added to the system')
    ax.plot(xgraph,ygraphearlycm248,label='Neutron Rate as a result of the Cm248 initially present in the system',c='brown')
    ax.legend(loc=0,prop={'size':6})
    ax.set_yscale('log')
    plt.xlabel("Years since February 3rd, 1995")
    plt.ylabel("Emitted Neutron Rate")
    
    
    
    

    def ChiSquareWithError0(obs,exp,error):
        chisq=0
        for i in range(len(obs)):
            chisq+=((obs[i]-exp[i])**2)/(error[i]**2)
        return chisq
    

    chisq=Curve([a,b])
    dof=len(ages)-len(guess)
    print(chisq,"chisq")
    #fig2,ax2=plt.subplots()
    ax.scatter(ages,neutronRates,color='red',s=1)
    plt.title("Projected Neutron Rate over Time. Chisq/dof="+str(round(chisq/dof,3)))
    ax.errorbar(ages,neutronRates,yerr=errors2,fmt='none',color='red')
    #ax2.set_yscale('log')
    plt.show()
    #print(max(ygraph))
    
    #scipyChisq=chisquare(neutronRates,ygraphatpoints,ddof=5)
    #print(scipyChisq)


if Version==1:
    print("Version 1 activating")
    def Curve1(guess):
        #x=[1.3589,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326]
        #neutronRates=np.array([8965,215.88,74.73,70.86,69.26,68.16,67.44,34.30])
        a=guess[0]
        f=guess[1]
        #f=(a/(decayConst252*mult252*chanFiss252))*(decayConst250*mult250*chanFiss250)
        b=guess[2]
    
        chisq=0

    
        for i in range(len(ages)):
            #WORKING#func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((decayConst248*chanAlpha252*a*ages[i])+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*ages[i]))+b)*np.exp(-decayConst248*ages[i])
            func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*ages[i]))*(decayConst248*mult248*chanFiss248)+b)*np.exp(-decayConst248*ages[i])
            chisq+=((neutronRates[i]-func)**2)/errors2[i]
        #print(chisq,"chisq")
        #print(guess)
        return chisq






    xgraph=np.linspace(0,30,200)
    guess=[10000,100,10]
    mintest=minimize(Curve1,guess,method='nelder-mead',options={'maxfev':9000000,'disp':True})
    #mintest=differential_evolution(Curve,bounds=[(0,10000000),(0.,1.),(0,10000000)],x0=guess)
    values=mintest['x']
    print(values,"values")

    

    print(chanAlpha252/(decayConst252*chanFiss252*mult252),"BIG TEST")



    Hfun = numdifftools.Hessian(Curve1, full_output="true")
    hessian_ndt, info = Hfun(values)
    
    se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
    print("Fit Uncertainty: " + str(se))

    
    
    
    a=values[0]
    
    f=values[1]
    #f=(a/(decayConst252*mult252*chanFiss252))*(decayConst250*mult250*chanFiss250)
    print(f,"f")
    b=values[2]
    cf252mass=a/(decayConst252*mult252*chanFiss252)
    cf250mass=f/(decayConst250*mult250*chanFiss250)
    cm248mass=b/(decayConst248*mult248*chanFiss248)
    print(cf252mass,cf250mass,cm248mass,"mass")
    
    cf252masserror=se[0]/(decayConst252*mult252*chanFiss252)
    cf250masserror=se[1]/(decayConst250*mult250*chanFiss250)
    cm248masserror=se[2]/(decayConst248*mult248*chanFiss248)
    print(cf252masserror,cf250masserror,cm248masserror,"masserror")
    
    
    print(100*cf252mass/(cf252mass+cf250mass+cm248mass),"mass percentage of cf252")
    print(100*cf250mass/(cf252mass+cf250mass+cm248mass),"mass percentage of cf250")
    print(100*cm248mass/(cf252mass+cf250mass+cm248mass),"mass percentage of cm248")

    cf252propmasserror=np.sqrt(((cf250mass+cm248mass)/((cf252mass+cf250mass+cm248mass)**2)*cf252masserror)**2+((-(cf252mass/(cf252mass+cf250mass+cm248mass)**2))*cf250masserror)**2+((-(cf252mass/(cf252mass+cf250mass+cm248mass)**2))*cm248masserror)**2)
    cf250propmasserror=np.sqrt(((cf252mass+cm248mass)/((cf250mass+cf252mass+cm248mass)**2)*cf250masserror)**2+((-(cf250mass/(cf250mass+cf252mass+cm248mass)**2))*cf252masserror)**2+((-(cf250mass/(cf250mass+cf252mass+cm248mass)**2))*cm248masserror)**2)
    
    
    cm248propmasserror=np.sqrt(((cf250mass+cf252mass)/((cf252mass+cf250mass+cm248mass)**2)*cm248masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf250masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf252masserror)**2)
    
    print(100*cf252propmasserror,"big equation252")
    print(100*cf250propmasserror,"big equation250")
    print(100*cm248propmasserror,"big equation248")
    
    normedprop=(values[0]/(decayConst252*chanFiss252))/(values[0]/(decayConst252*chanFiss252)+(1-values[0])/(decayConst250*chanFiss250))
    
    
    
    
 #WORKING#ygraph=(a*np.exp(-decayConst252*xgraph))+(f*np.exp(-decayConst250*xgraph))+((decayConst248*chanAlpha252*a*xgraph)+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*xgraph))+b)*np.exp(-decayConst248*xgraph)
    ygraph=(a*np.exp(-decayConst252*xgraph))+(f*np.exp(-decayConst250*xgraph))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)+b)*np.exp(-decayConst248*xgraph)
    
    ygraphcf252=a*np.exp(-decayConst252*xgraph)
    
    icpms250t0=(ygraphcf252[0]/(decayConst252*chanFiss252*mult252*8.164))*(decayConst250*chanFiss250*mult250)
    icpms250=icpms250t0*np.exp(-decayConst250*xgraph)
    ygraphcf250=f*np.exp(-decayConst250*xgraph)
    ygraphcm248init0=(chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)
    ygraphearlycm248=b*np.exp(-decayConst248*xgraph)
    
    print(min(ygraphcf252),max(ygraphcm248init0),(min(ygraphcf250)),"minmax")
    fig,ax=plt.subplots(dpi=250)
    #fig2,ax2=plt.subplots(dpi=250)
    #ax.set_xlim(20,28)
    #ax.set_ylim(-1,200)
    ax.plot(xgraph,ygraph,label='Total Neutron Rate')
    ax.plot(xgraph,ygraphcf252,label='Neutron Rate as a Result of Cf252')
    ax.plot(xgraph,ygraphcf250,label='Neutron Rate as a Result of Cf250')
    ax.plot(xgraph,ygraphcm248init0,label='Neutron Rate as a result of Cm248 being added to the system')
    #ax.plot(xgraph,icpms250,label="Neutron Rate of Cf250 if ICPMS was correct")
    ax.plot(xgraph,ygraphearlycm248,label='Neutron Rate as a result of the Cm248 initially present in the system')
    
    print(ygraphearlycm248)
    protohalflife=0
    halflife=[]
    for i in range(len(ygraph)-1):
        protohalflife=-(xgraph[i+1]-xgraph[i])*np.log(2)/(np.log((ygraph[i+1])/ygraph[i]))
        halflife.append(protohalflife)
    #print(halflife,"half life")
    xgraph2=np.linspace(0,30,199)
    #ax2.plot(xgraph2,halflife,label="Derivative test")
    ax.legend(loc=0,prop={'size':6})
    ax.set_yscale('log')
    plt.xlabel("Years since February 3rd, 1995")
    plt.ylabel("Emitted Neutron Rate")
    
    
    

    def ChiSquareWithError(obs,exp,error):
        chisq=0
        for i in range(len(obs)):
            chisq+=((obs[i]-exp[i])**2)/(error[i]**2)
        return chisq
    
    #This consideres the error with chisq
    #chisquarewitherror=(ChiSquareWithError(neutronRates,ygraphatpoints,errors2))
    #print(chisquarewitherror,"test2")
    #chisqdof=chisquarewitherror/5
    
    
    
    #plt.title("Projected Neutron Rate Over Time. Chisq/dof="+str(chisqdof))
    chisq=Curve1([a,f,b])
    dof=len(ages)-len(guess)
    print(chisq,"chisq")
    #fig2,ax2=plt.subplots()
    ax.scatter(ages,neutronRates,color='red',s=8)
    plt.title("Projected Neutron Rate over Time. Chisq/dof="+str(round(chisq/dof,3)))
    ax.errorbar(ages,neutronRates,yerr=errors2,fmt='none',color='red')
    

    #ax2.set_yscale('log')
    plt.show()
    #print(max(ygraph))
    
    #scipyChisq=chisquare(neutronRates,ygraphatpoints,ddof=5)
    #print(scipyChisq)

    
if Version==2:
    print("version 2 activating")
    print(ygraphearlycm248)
    def Curve2(guess):
        #x=[1.3589,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326]
        #neutronRates=np.array([8965,215.88,74.73,70.86,69.26,68.16,67.44,34.30])
        a=guess[0]
        #f=guess[1]
        f=guess[1]
           
        chisq=0

    
        for i in range(len(ages)):
            #WORKING#func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((decayConst248*chanAlpha252*a*ages[i])+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*ages[i]))+b)*np.exp(-decayConst248*ages[i])
            func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*ages[i]))*(decayConst248*mult248*chanFiss248))*np.exp(-decayConst248*ages[i])
            chisq+=((neutronRates[i]-func)**2)/errors2[i]
        #print(chisq,"chisq")
        #print(guess)
        return chisq






    xgraph=np.linspace(0,30,200)
    guess=[10000,100]
    mintest=minimize(Curve2,guess,method='nelder-mead',options={'maxfev':9000000,'disp':True})
    #mintest=differential_evolution(Curve,bounds=[(0,10000000),(0.,1.),(0,10000000)],x0=guess)
    values=mintest['x']
    
    
    
    
    Hfun = numdifftools.Hessian(Curve2, full_output="true")
    hessian_ndt, info = Hfun(values)
    
    se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
    print("Fit Uncertainty: " + str(se))
    

    print(values,"avlues")
    
    a=values[0]
    
    f=values[1]
    #f=(a/(decayConst252*mult252*chanFiss252*8.05))*(decayConst250*mult250*chanFiss250)
    print(f,"f")
    cf252mass=a/(decayConst252*mult252*chanFiss252)
    cf250mass=f/(decayConst250*mult250*chanFiss250)
    print(cf252mass,cf250mass)#,cm248mass,"mass")
    
    print((se[0]/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250))
    
    cf252masserror=se[0]/(decayConst252*mult252*chanFiss252)
    cf250masserror=se[1]/(decayConst250*mult250*chanFiss250)
    print(cf252masserror,cf250masserror,"masserror")
    
    
    print(100*cf252mass/(cf252mass+cf250mass),"mass percentage of cf252")
    print(100*cf250mass/(cf252mass+cf250mass),"mass percentage of cf250")

    cf252propmasserror=np.sqrt(((cf250mass)/((cf252mass+cf250mass)**2)*cf252masserror)**2+((-(cf252mass/(cf252mass+cf250mass)**2))*cf250masserror)**2+((-(cf252mass/(cf252mass+cf250mass)**2))*0)**2)
    cf250propmasserror=np.sqrt(((cf252mass)/((cf250mass+cf252mass)**2)*cf250masserror)**2+((-(cf250mass/(cf250mass+cf252mass)**2))*cf252masserror)**2+((-(cf250mass/(cf250mass+cf252mass)**2))*0)**2)
    
    
    
    print(100*cf252propmasserror,"big equation252")
    print(100*cf250propmasserror,"big equation250")
    
   # normedprop=(values[0]/(decayConst252*chanFiss252))/(values[0]/(decayConst252*chanFiss252)+(1-values[0])/(decayConst250*chanFiss250))
    
    
    
    
    #WORKING#ygraph=(a*np.exp(-decayConst252*xgraph))+(f*np.exp(-decayConst250*xgraph))+((decayConst248*chanAlpha252*a*xgraph)+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*xgraph))+b)*np.exp(-decayConst248*xgraph)
    ygraph=(a*np.exp(-decayConst252*xgraph))+(f*np.exp(-decayConst250*xgraph))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248))*np.exp(-decayConst248*xgraph)
    
    ygraphcf252=a*np.exp(-decayConst252*xgraph)
    ygraphcf250=f*np.exp(-decayConst250*xgraph)
    ygraphcm248init0=(chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)
    
    icpms250t0=(ygraphcf252[0]/(decayConst252*chanFiss252*mult252*8.164))*(decayConst250*chanFiss250*mult250)
    icpms250=icpms250t0*np.exp(-decayConst250*xgraph)
    
    fig,ax=plt.subplots(dpi=250)
    #ax.set_xlim(20,28)
    #ax.set_ylim(-1,200)
    ax.plot(xgraph,ygraph,label='Total Neutron Rate')
    ax.plot(xgraph,ygraphcf252,label='Neutron Rate as a Result of Cf252')
    ax.plot(xgraph,ygraphcf250,label='Neutron Rate as a Result of Cf250')
    ax.plot(xgraph,ygraphcm248init0,label='Neutron Rate as a result of Cm248 being added to the system')
    ax.plot(xgraph,icpms250,label="Neutron Rate of Cf250 if ICPMS was correct")
    #ax.plot(xgraph,(1400*np.exp(-xgraph*np.log(2)/5.17)),label="Extrapolation if avg. half life between 2016 and 2022")
    ax.legend(loc=0,prop={'size':6})
    ax.set_yscale('log')
    plt.xlabel("Year")
    plt.ylabel("Emitted Neutron Rate")
    ax.set_xticklabels([0,1995,2000,2005,2010,2015,2020,2025])
    
    
    

    def ChiSquareWithError(obs,exp,error):
        chisq=0
        for i in range(len(obs)):
            chisq+=((obs[i]-exp[i])**2)/(error[i]**2)
        return chisq
    
    #This consideres the error with chisq
    #chisquarewitherror=(ChiSquareWithError(neutronRates,ygraphatpoints,errors2))
    #print(chisquarewitherror,"test2")
    #chisqdof=chisquarewitherror/5
    
    
    
    #plt.title("Projected Neutron Rate Over Time. Chisq/dof="+str(chisqdof))
    chisq=Curve2([a,f])
    dof=len(ages)-len(guess)
    print(chisq,"chisq")
    #fig2,ax2=plt.subplots()
    ax.scatter(ages,neutronRates,color='red',s=8)
    plt.title("Projected Neutron Rate over Time. Chisq/dof="+str(round(chisq/dof,3)))
    ax.errorbar(ages,neutronRates,yerr=errors2,fmt='none',color='red')
    #ax2.set_yscale('log')
    plt.show()
    #print(max(ygraph))
    
    #scipyChisq=chisquare(neutronRates,ygraphatpoints,ddof=5)
    #print(scipyChisq)
    
    
if Version==3:
    print("Version 3 activating")
    def Curve3(guess):
        #x=[1.3589,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326]
        #neutronRates=np.array([8965,215.88,74.73,70.86,69.26,68.16,67.44,34.30])
        a=guess[0]
        #f=guess[1]
        f=(a/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)
        #b=guess[1]
    
        chisq=0

    
        for i in range(len(ages)):
            #func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+(chanAlpha252*((a*f/(chanFiss252*decayConst252*mult252))-(a*f/(chanFiss252*decayConst252*mult252))*np.exp(-decayConst252*ages[i]))*np.exp(-decayConst248*ages[i])))+b*np.exp(-decayConst248*ages[i])
            #########func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+((((a*f)/(chanFiss252*decayConst252*mult252))*decayConst248*chanAlpha252)*(ages[i]+(1/decayConst252)*np.exp(-decayConst252*ages[i]))+b)*(decayConst248*chanFiss248*mult248))
            #WORKING#func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((decayConst248*chanAlpha252*a*ages[i])+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*ages[i]))+b)*np.exp(-decayConst248*ages[i])
            func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*ages[i]))*(decayConst248*mult248*chanFiss248)+0)*np.exp(-decayConst248*ages[i])
            chisq+=((neutronRates[i]-func)**2)/errors2[i]
        #print(chisq,"chisq")
        #print(guess)
        return chisq






    xgraph=np.linspace(0,30,200)
    guess=[10000]
    mintest=minimize(Curve3,guess,method='nelder-mead',options={'maxfev':9000000,'disp':True})
    #mintest=differential_evolution(Curve,bounds=[(0,10000000),(0.,1.),(0,10000000)],x0=guess)
    values=mintest['x']
    
    
    
    
    Hfun = numdifftools.Hessian(Curve3, full_output="true")
    hessian_ndt, info = Hfun(values)
    
    se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
    print("Fit Uncertainty: " + str(se))
    

    print(values,"avlues")
    
    a=values[0]
    
    #f=values[1]
    f=(a/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)
    print(f,"f")
    #b=values[1]
    cf252mass=a/(decayConst252*mult252*chanFiss252)
    cf250mass=f/(decayConst250*mult250*chanFiss250)
    #cm248mass=b/(decayConst248*mult248*chanFiss248)
    print(cf252mass,cf250mass,"mass")
    
    print((se[0]/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250))
    
    cf252masserror=se[0]/(decayConst252*mult252*chanFiss252)
    cf250masserror=se[0]/(decayConst252*mult252*chanFiss252*8.164)
    #cm248masserror=se[1]/(decayConst248*mult248*chanFiss248)
    #print(cf252masserror,cf250masserror,cm248masserror,"masserror")
    
    
    print(100*cf252mass/(cf252mass+cf250mass),"mass percentage of cf252")
    print(100*cf250mass/(cf252mass+cf250mass),"mass percentage of cf250")
    #print(100*cm248mass/(cf252mass+cf250mass),"mass percentage of cm248")

    cf252propmasserror=np.sqrt(((cf250mass)/((cf252mass+cf250mass)**2)*cf252masserror)**2+((-(cf252mass/(cf252mass+cf250mass)**2))*cf250masserror)**2+((-(cf252mass/(cf252mass+cf250mass)**2))*0)**2)
    cf250propmasserror=np.sqrt(((cf252mass)/((cf250mass+cf252mass)**2)*cf250masserror)**2+((-(cf250mass/(cf250mass+cf252mass)**2))*cf252masserror)**2+((-(cf250mass/(cf250mass+cf252mass)**2))*0)**2)
    
    
    #cm248propmasserror=np.sqrt(((cf250mass+cf252mass)/((cf252mass+cf250mass+cm248mass)**2)*cm248masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf250masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf252masserror)**2)
    
    print(100*cf252propmasserror,"big equation252")
    print(100*cf250propmasserror,"big equation250")
    xgraph=np.linspace(0,30,200)
    ygraph=values[0]*np.exp(-decayConst252*xgraph)+(values[0]/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)*np.exp(-decayConst250*xgraph)+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(values[0]-values[0]*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)+0)*np.exp(-decayConst248*xgraph)
    ygraphcf252=values[0]*np.exp(-decayConst252*xgraph)
    ygraphcf250=(values[0]/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)*np.exp(-decayConst250*xgraph)
    ygraphcm248init0=((chanAlpha252/(decayConst252*chanFiss252*mult252))*(values[0]-values[0]*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)+0)*np.exp(-decayConst248*xgraph)
    normedprop=(values[0]/(decayConst252*chanFiss252))/(values[0]/(decayConst252*chanFiss252)+(1-values[0])/(decayConst250*chanFiss250))
    fig,ax=plt.subplots(dpi=250)
    #ax.set_xlim(20,28)
    #ax.set_ylim(-1,200)
    ax.plot(xgraph,ygraph,label='Total Neutron Rate')
    neutronRateGraph=ax.scatter(ages,neutronRates,color='red',s=10)
    ax.plot(xgraph,ygraphcf252,label='Neutron Rate as a Result of Cf252')
    ax.plot(xgraph,ygraphcf250,label='Neutron Rate as a Result of Cf250')
    ax.plot(xgraph,ygraphcm248init0,label='Neutron Rate as a result of Cm248 being added to the system')
    #ax.plot(xgraph,icpms250,label="Neutron Rate of Cf250 if ICPMS was correct")
    indexOfNow=min(range(len(xgraph)), key=lambda i: abs(xgraph[i]-27.8411))
    indexOfFirstDate=min(range(len(xgraph)), key=lambda i: abs(xgraph[i]-1.2849))
    indexOf2011Measurement=min(range(len(xgraph)), key=lambda i: abs(xgraph[i]-16.063))
    print(ygraph[indexOfNow],'Expected value of neutron rate after today')
    print(ygraph[indexOfFirstDate],"Possible earliest strength report")
    print(ygraph[indexOf2011Measurement],"Possible 2011 strength report")
    #If after today we get 8.3, then that's evidence that everything reported is right
    #If after today we get 15.8, then that's evidence of a mystery isotope
    ax.set_xticklabels([0,1995,2000,2005,2010,2015,2020,2025])
    plt.yscale('log')
    plt.legend(loc=0,prop={'size':6})
    chisq=Curve3([a,f])
    dof=len(ages)-len(guess)
    print(chisq,"chisq")
    #fig2,ax2=plt.subplots()
    ax.scatter(ages,neutronRates,color='red',s=1)
    plt.title("Projected Neutron Rate over Time. Chisq/dof="+str(round(chisq/dof,3)))
    ax.errorbar(ages,neutronRates,yerr=errors2,fmt='none',color='red')
    plt.show()
    
if Version==4:
    print("Version 4 activating")
    def Curve4(guess):
        #x=[1.3589,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326]
        #neutronRates=np.array([8965,215.88,74.73,70.86,69.26,68.16,67.44,34.30])
        a=guess[0]
        #f=guess[1]
        f=(a/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)
        b=guess[1]
    
        chisq=0

    
        for i in range(len(ages)):
            #func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+(chanAlpha252*((a*f/(chanFiss252*decayConst252*mult252))-(a*f/(chanFiss252*decayConst252*mult252))*np.exp(-decayConst252*ages[i]))*np.exp(-decayConst248*ages[i])))+b*np.exp(-decayConst248*ages[i])
            #########func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+((((a*f)/(chanFiss252*decayConst252*mult252))*decayConst248*chanAlpha252)*(ages[i]+(1/decayConst252)*np.exp(-decayConst252*ages[i]))+b)*(decayConst248*chanFiss248*mult248))
            #WORKING#func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((decayConst248*chanAlpha252*a*ages[i])+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*ages[i]))+b)*np.exp(-decayConst248*ages[i])
            func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*ages[i]))*(decayConst248*mult248*chanFiss248)+0)*np.exp(-decayConst248*ages[i])+b
            chisq+=((neutronRates[i]-func)**2)/errors2[i]
        #print(chisq,"chisq")
        #print(guess)
        return chisq






    xgraph=np.linspace(0,30,300)
    guess=[10000,10]
    mintest=minimize(Curve4,guess,method='nelder-mead',options={'maxfev':9000000,'disp':True})
    #mintest=differential_evolution(Curve,bounds=[(0,10000000),(0.,1.),(0,10000000)],x0=guess)
    values=mintest['x']
    
    
    
    
    Hfun = numdifftools.Hessian(Curve4, full_output="true")
    hessian_ndt, info = Hfun(values)
    
    se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
    print("Fit Uncertainty: " + str(se))
    

    print(values,"avlues")
    
    a=values[0]
    b=values[1]
    #f=values[1]
    f=(a/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)
    print(f,"f")
    #b=values[1]
    cf252mass=a/(decayConst252*mult252*chanFiss252)
    cf250mass=f/(decayConst250*mult250*chanFiss250)
    #cm248mass=b/(decayConst248*mult248*chanFiss248)
    print(cf252mass,cf250mass,"mass")
    
    print((se[0]/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250))
    
    cf252masserror=se[0]/(decayConst252*mult252*chanFiss252)
    cf250masserror=se[0]/(decayConst252*mult252*chanFiss252*8.164)
    #cm248masserror=se[1]/(decayConst248*mult248*chanFiss248)
    #print(cf252masserror,cf250masserror,cm248masserror,"masserror")
    
    
    print(100*cf252mass/(cf252mass+cf250mass),"mass percentage of cf252")
    print(100*cf250mass/(cf252mass+cf250mass),"mass percentage of cf250")
    #print(100*cm248mass/(cf252mass+cf250mass),"mass percentage of cm248")

    cf252propmasserror=np.sqrt(((cf250mass)/((cf252mass+cf250mass)**2)*cf252masserror)**2+((-(cf252mass/(cf252mass+cf250mass)**2))*cf250masserror)**2+((-(cf252mass/(cf252mass+cf250mass)**2))*0)**2)
    cf250propmasserror=np.sqrt(((cf252mass)/((cf250mass+cf252mass)**2)*cf250masserror)**2+((-(cf250mass/(cf250mass+cf252mass)**2))*cf252masserror)**2+((-(cf250mass/(cf250mass+cf252mass)**2))*0)**2)
    
    
    #cm248propmasserror=np.sqrt(((cf250mass+cf252mass)/((cf252mass+cf250mass+cm248mass)**2)*cm248masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf250masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf252masserror)**2)
    
    print(100*cf252propmasserror,"big equation252")
    print(100*cf250propmasserror,"big equation250")
    
    normedprop=(values[0]/(decayConst252*chanFiss252))/(values[0]/(decayConst252*chanFiss252)+(1-values[0])/(decayConst250*chanFiss250))
        
    
#WORKING#ygraph=(a*np.exp(-decayConst252*xgraph))+(f*np.exp(-decayConst250*xgraph))+((decayConst248*chanAlpha252*a*xgraph)+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*xgraph))+b)*np.exp(-decayConst248*xgraph)
    
    ygraph=(a*np.exp(-decayConst252*xgraph))+(f*np.exp(-decayConst250*xgraph))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)+0)*np.exp(-decayConst248*xgraph)+b
    
    ygraphcf252=a*np.exp(-decayConst252*xgraph)
    ygraphcf250=f*np.exp(-decayConst250*xgraph)
    ygraphcm248init0=(chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)
    #ygraphearlycm248=b*np.exp(-decayConst248*xgraph)
    mysteryIsotope=[b]*len(xgraph)
    
    fig,ax=plt.subplots(dpi=250)
    #ax.set_xlim(20,28)
    #ax.set_ylim(-1,200)
    totalNeutronRateGraph=ax.plot(xgraph,ygraph,label='Total Neutron Rate')
    cf252NeutronRateGraph=ax.plot(xgraph,ygraphcf252,label='Neutron Rate as a Result of Cf252')
    cf250NeutronRateGraph=ax.plot(xgraph,ygraphcf250,label='Neutron Rate as a Result of Cf250')
    cm248NeutronRateGraph=ax.plot(xgraph,ygraphcm248init0,label='Neutron Rate as a result of Cm248 being added to the system',color='red')
    mysteryIsotopeGraph=ax.plot(xgraph,mysteryIsotope,label='Rate due to mystery long lived isotope',color='magenta')
    neutronRateGraph=ax.scatter(ages,neutronRates,color='red',s=10)
    #ax.plot(xgraph,ygraphearlycm248,label='Neutron Rate as a result of the Cm248 initially present in the system')
    #ax.legend([totalNeutronRateGraph,cf252NeutronRateGraph,cf250NeutronRateGraph,cm248NeutronRateGraph,neutronRateGraph],["test1","test2","test3","test4","Measured Data"],scatterpoints=1,loc='upper right')

    ax.legend(loc=0,prop={'size':6})
    ax.set_xticklabels([0,1995,2000,2005,2010,2015,2020,2025])
    ax.set_yscale('log')
    plt.xlabel("Years since February 3rd, 1995")
    plt.ylabel("Emitted Neutron Rate")
    def ChiSquareWithError0(obs,exp,error):
        chisq=0
        for i in range(len(obs)):
            chisq+=((obs[i]-exp[i])**2)/(error[i]**2)
        return chisq

    chisq=Curve4([a,b])
    dof=len(ages)-len(guess)
    print(chisq,"chisq")
    #fig2,ax2=plt.subplots()
    plt.title("Projected Neutron Rate over Time. Chisq/dof="+str(round(chisq/dof,3)))
    ax.errorbar(ages,neutronRates,yerr=errors2,fmt='none',color='red')
    #ax2.set_yscale('log')
    plt.show()
    #print(max(ygraph))
    
    #scipyChisq=chisquare(neutronRates,ygraphatpoints,ddof=5)
    #print(scipyChisq)
    
if Version==5:
    print("Version 5 activating")
    def Curve5(guess):
        #x=[1.3589,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326]
        #neutronRates=np.array([8965,215.88,74.73,70.86,69.26,68.16,67.44,34.30])
        a=guess[0]
        #f=guess[1]
        f=(a/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)
        b=guess[1]
        #c=guess[2]
    
        chisq=0

    
        for i in range(len(ages)):
            #func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+(chanAlpha252*((a*f/(chanFiss252*decayConst252*mult252))-(a*f/(chanFiss252*decayConst252*mult252))*np.exp(-decayConst252*ages[i]))*np.exp(-decayConst248*ages[i])))+b*np.exp(-decayConst248*ages[i])
            #########func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+((((a*f)/(chanFiss252*decayConst252*mult252))*decayConst248*chanAlpha252)*(ages[i]+(1/decayConst252)*np.exp(-decayConst252*ages[i]))+b)*(decayConst248*chanFiss248*mult248))
            #WORKING#func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((decayConst248*chanAlpha252*a*ages[i])+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*ages[i]))+b)*np.exp(-decayConst248*ages[i])
            func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*ages[i]))*(decayConst248*mult248*chanFiss248)+0)*np.exp(-decayConst248*ages[i])+b*np.exp(-decayConstCo60*ages[i])
            #print(func,i,"test")
            chisq+=((neutronRates[i]-func)**2)/errors2[i]
        #print(chisq,"chisq")
        #print(guess)
        return chisq






    xgraph=np.linspace(0,30,200)
    guess=[10000,200]
    mintest=minimize(Curve5,guess,method='nelder-mead',options={'maxfev':9000000,'disp':True})
    #mintest=differential_evolution(Curve,bounds=[(0,10000000),(0.,1.),(0,10000000)],x0=guess)
    values=mintest['x']
    
    
    
    
    Hfun = numdifftools.Hessian(Curve5, full_output="true")
    hessian_ndt, info = Hfun(values)
    
    se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
    print("Fit Uncertainty: " + str(se))
    

    print(values,"avlues")
    
    a=values[0]
    
    #f=values[1]
    f=(a/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)
    print(f,"f")
    b=values[1]
    #c=values[2]
    cf252mass=a/(decayConst252*mult252*chanFiss252)
    cf250mass=f/(decayConst250*mult250*chanFiss250)
    #cm248mass=b/(decayConst248*mult248*chanFiss248)
    print(cf252mass,cf250mass,"mass")
    
    print((se[0]/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250))
    
    cf252masserror=se[0]/(decayConst252*mult252*chanFiss252)
    cf250masserror=se[0]/(decayConst252*mult252*chanFiss252*8.164)
    #cm248masserror=se[1]/(decayConst248*mult248*chanFiss248)
    #print(cf252masserror,cf250masserror,cm248masserror,"masserror")
    
    
    print(100*cf252mass/(cf252mass+cf250mass),"mass percentage of cf252")
    print(100*cf250mass/(cf252mass+cf250mass),"mass percentage of cf250")
    #print(100*cm248mass/(cf252mass+cf250mass),"mass percentage of cm248")

    cf252propmasserror=np.sqrt(((cf250mass)/((cf252mass+cf250mass)**2)*cf252masserror)**2+((-(cf252mass/(cf252mass+cf250mass)**2))*cf250masserror)**2+((-(cf252mass/(cf252mass+cf250mass)**2))*0)**2)
    cf250propmasserror=np.sqrt(((cf252mass)/((cf250mass+cf252mass)**2)*cf250masserror)**2+((-(cf250mass/(cf250mass+cf252mass)**2))*cf252masserror)**2+((-(cf250mass/(cf250mass+cf252mass)**2))*0)**2)
    
    
    #cm248propmasserror=np.sqrt(((cf250mass+cf252mass)/((cf252mass+cf250mass+cm248mass)**2)*cm248masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf250masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf252masserror)**2)
    
    print(100*cf252propmasserror,"big equation252")
    print(100*cf250propmasserror,"big equation250")
    xgraph=np.linspace(0,30,2000)
    ygraph=values[0]*np.exp(-decayConst252*xgraph)+(values[0]/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)*np.exp(-decayConst250*xgraph)+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(values[0]-values[0]*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)+0)*np.exp(-decayConst248*xgraph)+values[1]*np.exp(-decayConstCo60*xgraph)#+values[2]
    ygraphcf252=values[0]*np.exp(-decayConst252*xgraph)
    ygraphcf250=(values[0]/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)*np.exp(-decayConst250*xgraph)
    ygraphcm248init0=((chanAlpha252/(decayConst252*chanFiss252*mult252))*(values[0]-values[0]*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)+0)*np.exp(-decayConst248*xgraph)
    ygraphco60=values[1]*np.exp(-decayConstCo60*xgraph)
    normedprop=(values[0]/(decayConst252*chanFiss252))/(values[0]/(decayConst252*chanFiss252)+(1-values[0])/(decayConst250*chanFiss250))
    #steadyRate=[values[2]]*len(xgraph)
    fig,ax=plt.subplots(dpi=250)
    indexOfNow=min(range(len(xgraph)), key=lambda i: abs(xgraph[i]-27.8411))
    indexOfFirstDate=min(range(len(xgraph)), key=lambda i: abs(xgraph[i]-1.2849))
    indexOf2011Measurement=min(range(len(xgraph)), key=lambda i: abs(xgraph[i]-16.063))
   # print(values[2],"Neutron rate that is unaccounted for")
    print(ygraph[indexOfNow],'Expected value of neutron rate after today')
    print(ygraph[indexOfFirstDate],"Possible earliest strength report")
    print(ygraph[indexOf2011Measurement],"Possible 2011 strength report")
    def ChiSquareWithError0(obs,exp,error):
        chisq=0
        for i in range(len(obs)):
            chisq+=((obs[i]-exp[i])**2)/(error[i]**2)
        return chisq
    #ax.set_xlim(20,28)
    #ax.set_ylim(-1,200)
    ax.plot(xgraph,ygraph,label='Total Neutron Rate')
    neutronRateGraph=ax.scatter(ages,neutronRates,color='red',s=10)
    ax.plot(xgraph,ygraphcf252,label='Neutron Rate as a Result of Cf252')
    ax.plot(xgraph,ygraphcf250,label='Neutron Rate as a Result of Cf250')
    ax.plot(xgraph,ygraphcm248init0,label='Neutron Rate as a result of Cm248 being added to the system')
    #ax.plot(xgraph,steadyRate,label="Neutron Rate that is unaccounted for",c='magenta')
    ax.plot(xgraph,ygraphco60,label="Neutrons from Co60 decay")
    ax.set_xticklabels([0,1995,2000,2005,2010,2015,2020,2025])
    #ax.plot(xgraph,icpms250,label="Neutron Rate of Cf250 if ICPMS was correct")
    plt.yscale('log')
    plt.legend(loc=0,prop={'size':6})
    chisq=Curve5([a,b])
    dof=len(ages)-len(guess)
    print(chisq,"chisq")
    #fig2,ax2=plt.subplots()
    ax.scatter(ages,neutronRates,color='red',s=1)
    plt.title("Projected Neutron Rate over Time. Chisq/dof="+str(round(chisq/dof,3)))
    ax.errorbar(ages,neutronRates,yerr=errors2,fmt='none',color='red')
    plt.show()
    
if Version==6:
    print("Version 6 activating")
    def Curve6(guess):
        #x=[1.3589,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326]
        #neutronRates=np.array([8965,215.88,74.73,70.86,69.26,68.16,67.44,34.30])
        a=guess[0]
        #f=guess[1]
        f=guess[1]
        b=guess[1]
    
        chisq=0

    
        for i in range(len(ages)):
            #func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+(chanAlpha252*((a*f/(chanFiss252*decayConst252*mult252))-(a*f/(chanFiss252*decayConst252*mult252))*np.exp(-decayConst252*ages[i]))*np.exp(-decayConst248*ages[i])))+b*np.exp(-decayConst248*ages[i])
            #########func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+((((a*f)/(chanFiss252*decayConst252*mult252))*decayConst248*chanAlpha252)*(ages[i]+(1/decayConst252)*np.exp(-decayConst252*ages[i]))+b)*(decayConst248*chanFiss248*mult248))
            #WORKING#func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((decayConst248*chanAlpha252*a*ages[i])+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*ages[i]))+b)*np.exp(-decayConst248*ages[i])
            func=a*np.exp(-f*ages[i])
            chisq+=((neutronRates[i]-func)**2)/errors2[i]
        #print(chisq,"chisq")
        #print(guess)
        return chisq






    guess=[10000,1]
    mintest=minimize(Curve6,guess,method='nelder-mead',options={'maxfev':9000000,'disp':True})
    #mintest=differential_evolution(Curve,bounds=[(0,10000000),(0.,1.),(0,10000000)],x0=guess)
    values=mintest['x']
    
    print(values,"values")
    
    
    Hfun = numdifftools.Hessian(Curve6, full_output="true")
    hessian_ndt, info = Hfun(values)
    
    se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
    print("Fit Uncertainty: " + str(se))
    

    
    
    a=values[0]
    
    #f=values[1]
    f=values[1]
    print(se[0])
    
    print("The half life over the past few years is",np.log(2)/f,"+/-",np.abs((np.log(2)/f)-(np.log(2)/(f+se[1]))))
    print("Between",np.log(2)/(f-se[1]),"and",np.log(2)/(f+se[1]))
    
    #b=values[1]
    

    
    #cm248propmasserror=np.sqrt(((cf250mass+cf252mass)/((cf252mass+cf250mass+cm248mass)**2)*cm248masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf250masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf252masserror)**2)

    xgraph=np.linspace(0,30,1000)
    ygraph=values[0]*np.exp(-f*xgraph)
   
    fig,ax=plt.subplots(dpi=250)


    #ax.set_xlim(20,28)
    #ax.set_ylim(-1,200)
    ax.plot(xgraph,ygraph,label='Total Neutron Rate')
    neutronRateGraph=ax.scatter(ages,neutronRates,color='red',s=10)

    #ax.plot(xgraph,icpms250,label="Neutron Rate of Cf250 if ICPMS was correct")
    plt.yscale('log')
    plt.legend(loc=0,prop={'size':6})
    chisq=Curve6([a,f])
    dof=len(ages)-len(guess)
    print(chisq,"chisq")
    #fig2,ax2=plt.subplots()
    ax.scatter(ages,neutronRates,color='red',s=1)
    plt.title("Projected Neutron Rate over Time. Chisq/dof="+str(round(chisq/dof,3)))
    ax.set_xticklabels([0,1995,2000,2005,2010,2015,2020,2025])
    plt.ylabel("Emitted Neutron Rate (n/s)")
    plt.xlabel("Years since February 3rd, 1995")
    plt.xlim(20,30)
    plt.ylim(30,90)
    ax.errorbar(ages,neutronRates,yerr=errors2,fmt='none',color='green')
    plt.show()
    
if Version==7:
    print("Version 7 activating")
    def Curve7(guess):
        #x=[1.3589,16.063,21.4603,21.8438,21.9726,22.0932,22.1699,27.326]
        #neutronRates=np.array([8965,215.88,74.73,70.86,69.26,68.16,67.44,34.30])
        a=guess[0]
        #f=guess[1]
        f=(a/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)
        b=guess[1]
        c=guess[2]
        chisq=0

    
        for i in range(len(ages)):
            #func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+(chanAlpha252*((a*f/(chanFiss252*decayConst252*mult252))-(a*f/(chanFiss252*decayConst252*mult252))*np.exp(-decayConst252*ages[i]))*np.exp(-decayConst248*ages[i])))+b*np.exp(-decayConst248*ages[i])
            #########func=(a*((f*np.exp(-decayConst252*ages[i]))+(1-f)*(np.exp(-decayConst250*ages[i])))+((((a*f)/(chanFiss252*decayConst252*mult252))*decayConst248*chanAlpha252)*(ages[i]+(1/decayConst252)*np.exp(-decayConst252*ages[i]))+b)*(decayConst248*chanFiss248*mult248))
            #WORKING#func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((decayConst248*chanAlpha252*a*ages[i])+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*ages[i]))+b)*np.exp(-decayConst248*ages[i])
            func=(a*np.exp(-decayConst252*ages[i]))+(f*np.exp(-decayConst250*ages[i]))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*ages[i]))*(decayConst248*mult248*chanFiss248)+0)*np.exp(-decayConst248*ages[i])+b*np.exp(-c*ages[i])
            chisq+=((neutronRates[i]-func)**2)/errors2[i]
        #print(chisq,"chisq")
        #print(guess)
        return chisq






    xgraph=np.linspace(0,30,300)
    guess=[10000,10,0.1]
    mintest=minimize(Curve7,guess,method='nelder-mead',options={'maxfev':9000000,'disp':True})
    #mintest=differential_evolution(Curve,bounds=[(0,10000000),(0.,1.),(0,10000000)],x0=guess)
    values=mintest['x']
    
    
    
    
    Hfun = numdifftools.Hessian(Curve4, full_output="true")
    hessian_ndt, info = Hfun(values)
    
    se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
    print("Fit Uncertainty: " + str(se))
    

    print(values,"avlues")
    
    a=values[0]
    b=values[1]
    c=values[2]
    #f=values[1]
    f=(a/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250)
    print(f,"f")
    #b=values[1]
    cf252mass=a/(decayConst252*mult252*chanFiss252)
    cf250mass=f/(decayConst250*mult250*chanFiss250)
    #cm248mass=b/(decayConst248*mult248*chanFiss248)
    print(cf252mass,cf250mass,"mass")
    
    print((se[0]/(decayConst252*mult252*chanFiss252*8.164))*(decayConst250*mult250*chanFiss250))
    
    cf252masserror=se[0]/(decayConst252*mult252*chanFiss252)
    cf250masserror=se[0]/(decayConst252*mult252*chanFiss252*8.164)
    ax.set_xticklabels([0,1995,2000,2005,2010,2015,2020,2025])
    #cm248masserror=se[1]/(decayConst248*mult248*chanFiss248)
    #print(cf252masserror,cf250masserror,cm248masserror,"masserror")
    
    
    print(100*cf252mass/(cf252mass+cf250mass),"mass percentage of cf252")
    print(100*cf250mass/(cf252mass+cf250mass),"mass percentage of cf250")
    #print(100*cm248mass/(cf252mass+cf250mass),"mass percentage of cm248")

    cf252propmasserror=np.sqrt(((cf250mass)/((cf252mass+cf250mass)**2)*cf252masserror)**2+((-(cf252mass/(cf252mass+cf250mass)**2))*cf250masserror)**2+((-(cf252mass/(cf252mass+cf250mass)**2))*0)**2)
    cf250propmasserror=np.sqrt(((cf252mass)/((cf250mass+cf252mass)**2)*cf250masserror)**2+((-(cf250mass/(cf250mass+cf252mass)**2))*cf252masserror)**2+((-(cf250mass/(cf250mass+cf252mass)**2))*0)**2)
    
    
    #cm248propmasserror=np.sqrt(((cf250mass+cf252mass)/((cf252mass+cf250mass+cm248mass)**2)*cm248masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf250masserror)**2+((-(cm248mass/(cf252mass+cf250mass+cm248mass)**2))*cf252masserror)**2)
    
    
    normedprop=(values[0]/(decayConst252*chanFiss252))/(values[0]/(decayConst252*chanFiss252)+(1-values[0])/(decayConst250*chanFiss250))
        
    
#WORKING#ygraph=(a*np.exp(-decayConst252*xgraph))+(f*np.exp(-decayConst250*xgraph))+((decayConst248*chanAlpha252*a*xgraph)+((decayConst248*chanAlpha252*a/(decayConst252))*np.exp(-decayConst252*xgraph))+b)*np.exp(-decayConst248*xgraph)
    
    ygraph=(a*np.exp(-decayConst252*xgraph))+(f*np.exp(-decayConst250*xgraph))+((chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)+0)*np.exp(-decayConst248*xgraph)+b*np.exp(-c*xgraph)
    
    ygraphcf252=a*np.exp(-decayConst252*xgraph)
    ygraphcf250=f*np.exp(-decayConst250*xgraph)
    ygraphcm248init0=(chanAlpha252/(decayConst252*chanFiss252*mult252))*(a-a*np.exp(-decayConst252*xgraph))*(decayConst248*mult248*chanFiss248)
    #ygraphearlycm248=b*np.exp(-decayConst248*xgraph)
    mysteryIsotope=b*np.exp(-c*xgraph)
    fig,ax=plt.subplots(dpi=250)
    #ax.set_xlim(20,28)
    #ax.set_ylim(-1,200)
    totalNeutronRateGraph=ax.plot(xgraph,ygraph,label='Total Neutron Rate')
    cf252NeutronRateGraph=ax.plot(xgraph,ygraphcf252,label='Neutron Rate as a Result of Cf252')
    cf250NeutronRateGraph=ax.plot(xgraph,ygraphcf250,label='Neutron Rate as a Result of Cf250')
    cm248NeutronRateGraph=ax.plot(xgraph,ygraphcm248init0,label='Neutron Rate as a result of Cm248 being added to the system',color='red')
    mysteryIsotopeGraph=ax.plot(xgraph,mysteryIsotope,label='Rate due to mystery long lived isotope',color='magenta')
    neutronRateGraph=ax.scatter(ages,neutronRates,color='red',s=10)
    #ax.plot(xgraph,ygraphearlycm248,label='Neutron Rate as a result of the Cm248 initially present in the system')
    #ax.legend([totalNeutronRateGraph,cf252NeutronRateGraph,cf250NeutronRateGraph,cm248NeutronRateGraph,neutronRateGraph],["test1","test2","test3","test4","Measured Data"],scatterpoints=1,loc='upper right')

    ax.legend(loc=0,prop={'size':6})
    
    ax.set_yscale('log')
    plt.xlabel("Years since February 3rd, 1995")
    plt.ylabel("Emitted Neutron Rate")
    def ChiSquareWithError0(obs,exp,error):
        chisq=0
        for i in range(len(obs)):
            chisq+=((obs[i]-exp[i])**2)/(error[i]**2)
        return chisq

    chisq=Curve7([a,b,c])
    dof=len(ages)-len(guess)
    print(chisq,"chisq")
    print("The Half-Life of the mystery isotope is",np.log(2)/c,"years")
    #fig2,ax2=plt.subplots()
    plt.title("Projected Neutron Rate over Time. Chisq/dof="+str(round(chisq/dof,3)))
    ax.errorbar(ages,neutronRates,yerr=errors2,fmt='none',color='red')
    #ax2.set_yscale('log')
    plt.show()
    #print(max(ygraph))
    
    #scipyChisq=chisquare(neutronRates,ygraphatpoints,ddof=5)
    #print(scipyChisq)
    
    
end_time=time.time()-start_time
print("My program took",end_time,"to run!")