import numpy as np
import matplotlib.pyplot as plt


initialMass252=0.8
initialMass250=0.2
cf252MultiNoNorm=[0.026,0.1267,0.2734,0.3039,0.1848,0.0657,0.0154,0.0020,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
cf250MultiNoNorm=[0.03594,0.16771,0.29635,0.30104,0.14531,0.04740,0.00417,0.00208,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Cf252Multi=[x / sum(cf252MultiNoNorm) for x in cf252MultiNoNorm]
Cf250Multi=[x / sum(cf250MultiNoNorm) for x in cf250MultiNoNorm]
decayConst252=np.log(2)/2.65
decayConst250=np.log(2)/13.08
channelProportion252Alpha=0.96908
channelProportion252Fission=1-channelProportion252Alpha
channelProportion250Alpha=0.99923
channelProportion250Fission=1-channelProportion250Alpha
years=40
timeWindow=0.002#Time window in seconds,  might be increased to 0.002 in the future
multi252=3.735
multi250=3.50

time=np.linspace(0, years,years)


massprop252=initialMass252*np.exp(-decayConst252*time)/(initialMass250*np.exp(-decayConst250*time)+initialMass252*np.exp(-decayConst252*time))
massprop250=initialMass250*np.exp(-decayConst250*time)/(initialMass250*np.exp(-decayConst250*time)+initialMass252*np.exp(-decayConst252*time))

neutronprop252=(massprop252*multi252*decayConst252*channelProportion252Fission)/(massprop252*multi252*decayConst252*channelProportion252Fission+massprop250*multi250*decayConst250*channelProportion250Fission)
neutronprop250=(massprop250*multi250*decayConst250*channelProportion250Fission)/(massprop252*multi252*decayConst252*channelProportion252Fission+massprop250*multi250*decayConst250*channelProportion250Fission)
combinedmultiplicity=neutronprop252*multi252+neutronprop250*multi250


#mass252=initialMass252*np.exp(-decayConst252*time)*multi252*decayConst252*channelProportion252Fission
#mass250=initialMass250*np.exp(-decayConst250*time)*multi250*decayConst250*channelProportion250Fission
mass252=initialMass252*np.exp(-decayConst252*time)
mass250=initialMass250*np.exp(-decayConst250*time)
totalmass=mass252+mass250

neutrons252=mass252*multi252*decayConst252*channelProportion252Fission


decayConst=[]
for i in range(len(totalmass)-1):
    negativeslope=(-1/(time[i+1]-time[i]))*(np.log(totalmass[i+1]/totalmass[i]))
    decayConst.append(negativeslope)

halflife=np.log(2)/decayConst
print(halflife)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

plt.title("Mass Proportion of Cf252 and Cf250")
plt.xlabel("Time (Years)")
plt.ylabel("Proportion")
plt.plot(massprop252,label="Mass Proportion of Cf252")
plt.plot(massprop250,label="Mass Proportion of Cf250")
plt.legend()
plt.show()

plt.title("Half Life over Time")
plt.xlabel("Time (Years)")
plt.ylabel("Half Life (Years)")
plt.plot(halflife)
plt.legend()
plt.show()

fig2=plt.figure()
ax2=fig.add_subplot(2,2,2)
plt.title("Neutron Multiplicity over time")
plt.xlabel("Time (Years)")
plt.ylabel("Average Multiplicity (# neutrons)")
plt.plot(combinedmultiplicity)

interestedInMicroPlots=False
if(interestedInMicroPlots):
    fig3=plt.figure()
    ax3=fig.add_subplot(3,3,3)
    plt.title("Occurance of each bin over time")
    plt.ylabel("Probability of a bin occuring")
    plt.xlabel("Time (Years)")
    bin0=neutronprop252*Cf252Multi[0]+neutronprop250*Cf250Multi[0]
    bin1=neutronprop252*Cf252Multi[1]+neutronprop250*Cf250Multi[1]
    bin2=neutronprop252*Cf252Multi[2]+neutronprop250*Cf250Multi[2]
    bin3=neutronprop252*Cf252Multi[3]+neutronprop250*Cf250Multi[3]
    bin4=neutronprop252*Cf252Multi[4]+neutronprop250*Cf250Multi[4]
    bin5=neutronprop252*Cf252Multi[5]+neutronprop250*Cf250Multi[5]
    bin6=neutronprop252*Cf252Multi[6]+neutronprop250*Cf250Multi[6]
    bin7=neutronprop252*Cf252Multi[7]+neutronprop250*Cf250Multi[7]
    bin8=neutronprop252*Cf252Multi[8]+neutronprop250*Cf250Multi[8]
    plt.plot(bin0,label="bin1")
    plt.legend()
    plt.show()
    plt.plot(bin1,label="bin2")
    plt.legend()
    plt.show()
    plt.plot(bin2,label="bin3")
    plt.legend()
    plt.show()
    plt.plot(bin3,label="bin4")
    plt.legend()
    plt.show()
    plt.plot(bin4,label="bin5")
    plt.legend()
    plt.show()
    plt.plot(bin5,label="bin6")
    plt.legend()
    plt.show()
    plt.plot(bin6,label="bin7")
    plt.legend()
    plt.show()
    plt.plot(bin7,label="bin8")
    plt.legend()
    plt.show()
    plt.plot(bin8,label="bin9")
    plt.legend()
    plt.show()


r=np.linspace(0,1000,1000)
tw=0.002
double=[]


