import DateFitter as DateFitter
import DataGenerator_ColinMethod as DataGenerator
import numpy as np
import MLE_No_Curium_Refactor as ProportionFitter
import Shape_Fitter
fulldata,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10=[],[],[],[],[],[],[],[],[],[],[]

#Syntax for Data Generator:
    #DataGenerator.DataGenerator(Years=[],protoFissions=mass integer,Eff=Efficiency,Cf252Prop = Proportion of Cf252 by mass,Cf250Prop = Proportion of Cf252 by mass,RunLength = Length of Run in seconds,doTheMLE =Boolean, Set this to False)
    #Example: DataGenerator.DataGenerator([12,13,14,15,16],10000000,0.47,0.85,0.15,[14400,14400,14400,14400,28800],False)

#Syntax for Date Fitter:
    #DateFitter.DataFitter(data=[[]], Cf252= float proportion of Cf252, Cf250= float proportion of Cf250,guess=[guess of age of first sample, guess of initial mass, guess of efficiency] )
    #Example: DateFitter.DataFitter(np.concatenate([data1,data2,data3,data4]),0.85,0.15,[12,12000000,0.45])
  
#Syntax for Proportion Fitter:
    #ProportionFitter.ProportionFitter(data=[[]],yearOfInterest=[] Age of each of the lists in the data list of lists (years), runLength=[] Length of the runs in seconds of each run in the data list of lists)
    #Example: ProportionFitter.ProportionFitter(np.concatenate([data1,data2,data3,data4]),[12,13,14,15],[14400,14400,14400,28800])


#Mode 0 is for data generation, mode 1 is for year fitting, mode 2 is for proportion fitting
mode=3


if mode==0:
    
    DataGenerator.DataGenerator([40],1000000000,0.481,0.85,0.15,[14400],False)

if mode==1:
    realDataColin,realData0,realData1,realData2,realData3,realData4=[],[],[],[],[],[]
    realDataColin.append(np.genfromtxt('4860.csv',delimiter=',')),realData0.append(np.genfromtxt('5664.csv',delimiter=',')),realData1.append(np.genfromtxt('5719.csv',delimiter=',')),realData2.append(np.genfromtxt('5768.csv',delimiter=',')),realData3.append(np.genfromtxt('5807.csv',delimiter=',')),realData4.append(np.genfromtxt('8197.csv',delimiter=','))
    #realData0.append(np.genfromtxt('TestDataFissions11.csv',delimiter=',')),realData1.append(np.genfromtxt('TestDataFissions12.csv',delimiter=',')),realData2.append(np.genfromtxt('TestDataFissions13.csv',delimiter=',')),realData3.append(np.genfromtxt('TestDataFissions14.csv',delimiter=',')),realData4.append(np.genfromtxt('TestDataFissions15.csv',delimiter=','))
    
    datae=([realData0,realData1,realData2,realData3,realData4])
    data=np.concatenate(datae)
    yearGuess=[13]#,13,14,15]
    initialAmountGuess=10000000
    EffGuess=0.47
    Cf252=0.85
    Cf250=0.15
    guess=[yearGuess[0],initialAmountGuess,EffGuess]
    #x=[75,77.5,80,81,82.5,85,86,87.5,90,92.5]
    #y=[36.27,36.93,37.62,37.93,38.42,38.59,39.30,39.27,38.85,38.23,37.68]
    DateFitter.DateFitter(np.concatenate([realData0,realData1,realData2,realData3,realData4]),Cf252,Cf250,[20,10000000,0.48],[14402,14402,14402,14402,25786])
    #DateFitter.DateFitter(np.concatenate([realData0,realData1,realData2,realData3]),Cf252,Cf250,[20,10000000,0.48],[14402,14402,14402,14402])

    
    
if mode==2:
    yearOfInterest=[33.932,34.06,34.178,34.255,39.4137]
    time=np.linspace(0,np.round(max(yearOfInterest)),1000)
     
    for i in range(len(yearOfInterest)):
        fulldata.append(np.genfromtxt(('TestDataFissions'+str(yearOfInterest[i])+'.csv'),delimiter=','))
    
    runLength=[14400,14400,14400,14400,25786]
        
    ##datae=np.array([dataFromGeneratorsame3,dataFromGeneratorsame4,dataFromGeneratorsame7,dataFromGeneratorsame5,dataWithNewFit5])
     
    print(len(fulldata),"testlength")
    print(len(yearOfInterest),"testyearlength")
    data=np.array(fulldata)
    ProportionFitter.ProportionFitter(np.array(fulldata),yearOfInterest,runLength)

if mode==3:
   # yearGuess=[21.2301]
   #This was computed by taking the years between February 3rd 1995 and July 15th 2016
    yearGuess=[21.4603]
    initialAmountGuess=40000000
    EffGuess=0.48
    Cf252Guess=0.5
    Cf250Guess=2000.
    #Cm248Guess=.1
    MysteryIsotopeGuess=20
    #Cf250=0.15
    guess=[yearGuess[0],initialAmountGuess,EffGuess,Cf252Guess,MysteryIsotopeGuess]#,0,0.1,0.2,0.3,0.2,0.1,0.05,0.05] #0.0261, 0.127, 0.274, 0.3045, 0.1852, 0.0658, 0.0154, 0.002]
    #RunLengths=[14402,14402,14402,14402,25786]
    RunLengths=[4801,14402,14402,14402,14402,25786,22334]
    #backgroundRate=[0.075,0.0671,0.0671,0.0671,0.0671,0.11]
    backgroundRate=[0.02,0.02,0.02,0.02,0.02,0.02,0.02]
    #backgroundRate=[0.01505,0.01505,0.01505,0.01505,0.01505]
    realData0,realData1,realData2,realData3,realData4,realData5,realData6,realData7,realData8,realData9,realData10,realData11,realData12,realData13,realData14=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    realData0.append(np.genfromtxt('4860.csv',delimiter=',')),realData1.append(np.genfromtxt('5384.csv',delimiter=',')),realData2.append(np.genfromtxt('5664.csv',delimiter=',')),realData3.append(np.genfromtxt('5719.csv',delimiter=',')),realData4.append(np.genfromtxt('5768.csv',delimiter=',')),realData5.append(np.genfromtxt('5807.csv',delimiter=',')),realData6.append(np.genfromtxt('8197.csv',delimiter=',')),realData7.append(np.genfromtxt('8411.csv',delimiter=','))#,realData6.append(np.genfromtxt('TestDataFissions27.8.csv',delimiter=','))#,realData5.append(np.genfromtxt('TestDataFissions39.6.csv',delimiter=','))
    #     655
    #Note that this is ignoring the run 4860. This is because the conditions were slightly different for this run.
    datae=([realData1,realData2,realData3,realData4,realData5,realData6,realData7])#,realData5,realData6,realData7,realData8,realData9,realData10,realData11,realData12,realData13,realData14])
    data=np.concatenate(datae)
    values_of_fit_parameters=Shape_Fitter.DateFitter(data,guess,RunLengths,backgroundRate)
    #print(values_of_fit_parameters,"fit parameters in another program")
    #errorComputer=Shape_Fitter.Errorcomputer(values_of_fit_parameters)
    # ==========================================================
    