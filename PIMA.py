#Code for computing axion mass vs Hubble friction after inflation in the DAMP models


# Would like to evolve from the end of inflation until the thermalization of the higgs fields

#PSEUDOCODE

#Ch 0: Initialization

#0.1: Define initial conditions
#0.2: Define functional for axion mass in terms of Lambda, faDyn, LambdaPhys, T functions
#0.3: Define functions for Lambda, faDyn, LambdaPhys, T including all relevant scaling, and include effects of additional fermion pairs in terms of some Npsi
#0.4: Define meshgrid for HIxTR
#0.5 Define meshgrid for output, ma/H_max

#Ch 1: Cosmology Evolution(of a given TR, HI)

#I think to evalute the results, to cut down on computing time we do the following:
	#For any point we are evaluating, we first check whether the inflationary condition is met i.e. ma_0/H_I > 1
	#Then, if that passes, we can begin evaluating from T_max until T_therm (rho_H = T_H^4). Once H<HBmu we continute to evalute until H_therm OR ma/H > 1
	#As we evaluate, we write each point to a temporary array and at the end of evaluation, we keep the max with the np.ndarray.max function
	#The last value that was kept can be written to an output meshgrid to be plotted visually with a heatmap/contour plot
#I was going to define various functions for this part, but it seems there is too much feedback to make this part modular, so I've define a single function which does the entire job in the following way:
	#1.1 First we write a simple function which evaluates the scatter rate of the higgs from the gluon loop at a given temperature and field value of Phi
	#1.2 Now we need to think about what values of H we might scan over. The absolute limits of evaluation, in terms of a fiducial temperature given by Rho_I or Rho_R in MD or RD eras respectively (i.e. not accounting for the higgs radiation). Since H ~ T^4 in MD and H ~ T^2 in RD, we can find the number of e-folds in H by counting the e-folds in T from T_max to TR_min then write a logspace array with 100 values in each H e-fold
	#1.3 We define some initial conditions to compare H ~ Gamma at HI then do while until H < Gamma and the Higgs has thermalized
	
#1.4 During the previous while loop, we can evaluate the axion mass and break the loop under appropriate conditions



#Ch 2: Evaluation
#2.0 Write loop to scan over different values of y (begin with just one value y_0 = 10^-6)
#2.1 Write loop to scan over HIxTR meshgrid
#2.2 For each point, use cosmology functions to find max ma/H 
#2.3 Write each of these points to the output meshgrid during the HIxTR meshgrid scane
#2.4 For new values of Y, we should write additional meshgrids

# We can find the maximum allowed parameter space in a scan over y's in the following way:
	#For each meshgrid in the y scan, we take binary values for ma/H >1, <1 during the critical era, and sum over the entire grid
	#This way, the grid with the most allowed points has the smallest sum, and we plot that one with its y value

#2.5 Binary sum over y meshgrids, minimize the sum
#2.6 Plot using a contour map the minimzed meshgrid


#CODE: FIRST SEQUENCE

#Ch 0: Initialization
import numpy as np
import matplotlib.pyplot as plt

#0.1: Define initial conditions
def Constants():
	global Mplank
	global gstar 
	global kgamma

	Mplank = 2.4e18
	gstar = 228.
	kgamma = ((np.pi**2)*gstar)/30.

def Parameters():
	global Mplank
	global Msusy
	global Mgut 
	global Bmu 
	global fa
	global phi_i
	global c
	global L_0

	Msusy = 1.e3
	Mplank = 2.4e18
	Mgut = 2.e16
	Bmu = (1/50.)*1.e6
	fa = 2e9
	phi_i = 2.e16
	Bmu = (1/50.)*1.e6
	c = (Mplank/Mgut)**2
	L_0 = 1.3e7

def GridInit():
	global HI_max
	global HI_min
	global TR_max
	global TR_min
	global L_max
	global L_min

	HI_max = 1.e7
	HI_min = 1.e5
	TR_max = 1.e12
	TR_min = 1.e1
	L_max  = 1.e13
	L_min  = 1.e12

#functions for scaling parameters

#At the start of a cosmology evaluation, HI, TR, L_i, phi_i, y are presumably defined, as well as H (1/H is our time) by menans of a np.logspace generated array
def PhiMD(H,HI):
	Parameters()
	phi = phi_i*(H/HI)
	return phi

def PhiRD(H,HI,TR):
	Constants()
	Parameters()
	HTR = ((np.sqrt(kgamma/3.))*(TR**2)/Mplank)
	phi = phi_i*(HTR/HI)*((H/HTR)**(3./4))
	return phi

def PhiAny(H,HI,TR):	
	HTR = ((np.sqrt(kgamma/3.))*(TR**2)/Mplank)
	if(H > HTR):
		phi = PhiMD(H,HI)
	elif(H <= HTR):
		phi = PhiRD(H,HI,TR)
	return phi

def Pvev1(H):
	Constants()
	P1 = np.sqrt(H*Mplank)
	return P1

def Pvev2(L_i):
	Parameters()
	P2 = ((L_i**6)*(Mgut**2)/(fa**(2./3)))**(3./22)
	return P2


#up until here, everything can be calculated with I.C.

#next are Qvevs, which depend on phi & P
def Qvev1(H,L_i,y,P,phi):
	#VEV from cH^2 P->fa0?
	Parameters()
	#Q = (((L_i**9)*((phi/phi_i)**6))/(c*(H**2)*P*y))**(1/6.)
	#Raymond's formula
	Q = (((L_i**9)*((phi/phi_i)**6))/(c*(H**2)*fa*y))**(1/6.)
	return Q

def Qvev2(L_i,P,y,phi):
	#VEV from y^2P^2  raymond says P->fa0 from lambda~, but P in yp^2 still P??
	Parameters()
	#Q = (((L_i**9)*((phi/phi_i)**6))/((y*P)**3))**(1/6.)
	#raymond's formula
	Q = (((L_i**9)*((phi/phi_i)**6))/((y**3)*(fa)*(P**2)))**(1/6.)
	
	return Q

def Qvev3(L_i,P,y,phi):
	#VEV from y^2Q^4 P->fa0?
	Parameters()
	#Q = (((L_i**9)*((phi/phi_i)**6))/((y**3)*P))**(1/8.)
	#Raymond's formula
	Q = (((L_i**9)*((phi/phi_i)**6))/((y**3)*fa))**(1/8.)
	
	return Q

def Qmin(H,L_i,y,P,phi):
	Q1 = Qvev1(H,L_i,y,P,phi)
	Q2 = Qvev2(L_i,P,y,phi)
	Q3 = Qvev3(L_i,P,y,phi)
	Qm = np.amin([Q1,Q2,Q3])

	return Qm


def PvevQ(H,L_i,y,P,phi):
	Parameters()
	Q1 = Qvev1(H,L_i,y,P,phi)
	Q2 = Qvev3(L_i,P,y,phi)
	Qm = np.amin([Q1,Q2])

	myPvev = (Mgut**(2./5))*(y**(1./5))*(((L_i**9)*((phi/phi_i)**6)*(1/(fa*y)))**(1./10))/(Qm**(1./5))

	return myPvev

def Pmax(H,L_i,y,phi):
	Parameters()
	P1 = Pvev1(H)
	P2 = Pvev2(L_i)
	P3 = PvevQ(H,L_i,y,fa,phi)
	Pm = np.amax([fa,P1,P2,P3])
	return Pm

def faDyn(H,L_i,y,phi):
	ThisP    = Pmax(H,L_i,y,phi)
	ThisQ    = Qmin(H,L_i,y,ThisP,phi)

	faEff = np.amax([ThisP,ThisQ])
	return faEff, ThisQ, ThisP
#up until now, the standard computational chain is compute phi -> compute P -> compute Q -> compute fa 
#as far as I can tell, I have not defined the functions in anyway as to compute things twice if I first use PhiAny then faDyn(PhiAny)

def Lambda(L_i,phi,Npsi,P):
	Parameters()
	L1 = L_i*((phi/phi_i)**((2./3)+(2*Npsi/9.)))*((P/fa)**(1./9))
	L0 = L_0*((phi/phi_i)**(2./3))*((P/fa)**(1./9))
	return np.amax([L1,L0])

def LambdaTilde(L,y,P,phi):
	Parameters()
	LTilde = (((L**9.)/(y*fa))**(1./8))*((phi/phi_i)**(3./4))
	return LTilde

def LambdaPhys(L,y,P):
	Parameters()
	LPhys = L*np.amax([1.,(L/(y*fa))**(1./8)])
	return LPhys

def TemperatureMD(H,TR):
	Constants()
	T = ((TR**2)*H*Mplank*(np.sqrt(3./kgamma)))**(1./4)
	return T

def TemperatureRD(H):
	Constants()
	T = (H*Mplank*(np.sqrt(3./kgamma)))**(1./2)
	return T

def FidT(H,TR):
	if (TemperatureMD(H,TR) > TR):
		T = TemperatureMD(H,TR)
	else:
		T = TemperatureRD(H)
	return T

def THiggs(H,phi): #fix scaling during RD
	Constants()
	Parameters()
	Th = ((Msusy**2.)*phi*(6.19e-5)/(H*kgamma))**(1./2)
	return Th

def GammaH(H,T,phi):
	Gamma = (6.19e-5)*(T**2)/phi
	return Gamma

def maZT(L,faTrue,y,Q,P):
	Parameters()

	if((y*P) > Q):
		ma = (L**(3./2))*(Msusy**(1./2))/(faTrue)
	else:
		Ltilde = LambdaTilde(L,y,P,phi_i)
		ma = (Ltilde**(3./2))*(Msusy**(1./2))/(faTrue)*np.sqrt(Ltilde/Q)
		
	return ma

def maFT(L,LPhys,faTrue,T,Npsi,y,Q,P,phi):	#two cases for Q ? yP
	Parameters()
	if((y*P) > Q):
		ma = (L**(3./2))*(Msusy**(1./2))/(faTrue)*(np.amin([((LPhys/T)**3.),1]))
	else:
		Ltilde = LambdaTilde(L,y,P,phi)
		ma = (np.sqrt(Ltilde/Q))*(Ltilde**(3./2))*(Msusy**(1./2))/(faTrue)*(np.amin([((LPhys/T)**3.),1]))
	return ma



#0.4: Define meshgrid for HIxTR
#0.5: Define meshgrid for output, ma/H_max

def HIxTR(HI_Max,HI_Min,TR_Max,TR_Min):

	GridInit()
	HI_exp = np.log10(HI_Max) - np.log10(HI_Min)
	TR_exp = np.log10(TR_Max) - np.log10(TR_Min)

	HI = np.logspace(np.log10(HI_Min),np.log10(HI_Max),(HI_exp*50))
	TR = np.logspace(np.log10(TR_Min),np.log10(TR_Max),(TR_exp*50))
	HH,TT = np.meshgrid(HI,TR)
	
	maByHMax = np.zeros([np.shape(HH)[0],np.shape(HH)[1]])

	return HH,TT,maByHMax






#Ch 1: Cosmology (of a given TR, HI)

#I was going to define various functions for this part, but it seems there is too much feedback to make this part modular, so I've define a single function which does the entire job in the following way:

def CosmoEvol(HI,TR,y,L_i,Npsi):
	Constants()
	Parameters()

#1.2 Now we need to think about what values of H we might scan over. The absolute limits of evaluation, in terms of a fiducial temperature given by Rho_I or Rho_R in MD or RD eras respectively (i.e. not accounting for the higgs radiation). Since H ~ T^4 in MD and H ~ T^2 in RD, we can find the number of e-folds in H by counting the e-folds in T from T_max to TR_min then write a logspace array with 100 values in each H e-fold

	Tmax = np.sqrt(np.sqrt(HI*Mplank)*TR)
	Tmin = (10**(np.log10(TR)-2))
	exp  = 4*(np.log10(Tmax)-np.log10(TR))+2*(np.log10(TR)-np.log10(Tmin))
	Hmin = HI*(10**(-exp))

	H	     = np.logspace(np.log10(HI),np.log10(Hmin),(exp*50))
	maHRatio     = []
	n	     = 0
	HBmuNoOsc    = True
	HBmu	     = 0.
	HPreBmuTherm = False
	Htherm       = HI

#1.3 We define some initial conditions to compare H ~ Gamma at HI then do while until H < Gamma and the Higgs has thermalized
	myPhi = phi_i
	myH   = H[0]
	myT   = np.amax([THiggs(myH,myPhi),FidT(myH,TR)])
#1.3.1 Before Starting the loop, we check to see if the axion mass is large enough such that the field oscillates during inflation, if not the point is excluded
	myFa,myQ,myP	     = faDyn(myH,L_i,y,myPhi)
	maHIRatio	     = (maZT(L_i,myFa,y,myQ,myP)/(3*myH))
	InflationSuppression = (maHIRatio > 1.)
#Fix axion zero temp mass
	if(InflationSuppression):
		while(myH > GammaH(myH,myT,myPhi)):	
			n+=1
			myH   = H[n]
			myPhi = PhiAny(myH,HI,TR)
			myT   = np.amax([THiggs(myH,myPhi),FidT(myH,TR)])

			Htherm = myH

			if(c*(H[n]**2) < Bmu and HBmuNoOsc):
				HBmu = H[n]
				HBmuNoOsc = False

			if(not HBmuNoOsc):
				myFa,myQ,myP  = faDyn(myH,L_i,y,myPhi)
				myL	      = Lambda(L_i,myPhi,Npsi,myP)
				myLP	      = LambdaPhys(myL,y,myP)
				ma 	      = maFT(myL,myLP,myFa,myT,Npsi,y,myQ,myP,myPhi)

				maHRatio.append(ma/(3*H[n]))



		if(HBmu < Htherm):
			maHRatio.append(0.)
			HPreBmuTherm = True
	
	elif(not InflationSuppression):	
		maHRatio.append(1.1)
		
	maHRatioMax = np.amax(maHRatio)

	return maHRatioMax, InflationSuppression, HBmuNoOsc, HPreBmuTherm


def CosmoChk(HI,TR,y,L_i,Npsi,PlotBool):
	Constants()
	Parameters()

	Tmax = np.sqrt(np.sqrt(HI*Mplank)*TR)
	Tmin = (10**(np.log10(TR)-2))
	exp  = 4*(np.log10(Tmax)-np.log10(TR))+2*(np.log10(TR)-np.log10(Tmin))
	Hmin = HI*(10**(-exp))

	H	      = np.logspace(np.log10(HI),np.log10(Hmin),(exp*50))
	maHRatio      = np.zeros(np.size(H))
	Temperature   = np.zeros(np.size(H))
	LambdaTracker = np.zeros(np.size(H))

	n      = 0
	nBmu   = np.size(H) - 1
	nTherm = 0

	HThermBool   = False
	HBmuNoOsc    = True
	HPreBmuTherm = False
	myPhi = phi_i
	myH   = H[0]
	myT   = np.amax([THiggs(myH,myPhi),FidT(myH,TR)])
#1.3.1 Before Starting the loop, we check to see if the axion mass is large enough such that the field oscillates during inflation, if not the point is excluded
	myFa,myQ,myP	     = faDyn(myH,L_i,y,myPhi)
	maHIRatio	     = (maZT(L_i,myFa,y,myQ,myP)/(3*myH))
	maHRatio[0]	     = maHIRatio	
	InflationSuppression = (maHIRatio > 1.)
#Fix axion zero temp mass
	if(InflationSuppression):
		for n in range(len(H)):
			myH   = H[n]
			myPhi = PhiAny(myH,HI,TR)

			if(not HThermBool):
				myT   = np.amax([THiggs(myH,myPhi),FidT(myH,TR)])
			elif(HThermBool):
				myT   = FidT(myH,TR)

			myFa,myQ,myP  = faDyn(myH,L_i,y,myPhi)
			myL	      = Lambda(L_i,myPhi,Npsi,myP)
			myLP	      = LambdaPhys(myL,y,myP)
			ma 	      = maFT(myL,myLP,myFa,myT,Npsi,y,myQ,myP,myPhi)

			maHRatio[n]      = (ma/(3*H[n]))
			Temperature[n]   = myT	
			LambdaTracker[n] = myL

			if(myH < GammaH(myH,myT,myPhi) and (not HThermBool)):
				nTherm = n
				Htherm = myH
				HThermBool = True			 
			if(c*(myH**2) < Bmu and HBmuNoOsc):
				nBmu = n
				HBmu = myH
				HBmuNoOsc = False

	
	elif(not InflationSuppression):	
		maHRatio[:] = maHIRatio
		
	HBmu = H[nBmu]
	Htherm = H[nTherm]

	if(PlotBool):
		ymin=np.amin(maHRatio)
		ymax=np.amax(maHRatio)
		yline=np.logspace(np.log10(ymin),np.log10(ymax),np.size(H))
		xBmu = np.copy(yline)
		xBmu[:] = 1/HBmu
		xTherm = np.copy(yline)
		xTherm[:] = 1/Htherm


		plt.figure(1)
		plt.plot((1/H),maHRatio,label='$ma/3H$')
		plt.plot(xBmu,yline,linestyle='-',label='$H_{B\mu}$')
		plt.plot(xTherm,yline,linestyle='-.',label='$H_{th}$')
		plt.plot((1/H),Temperature,label='$T$')
		plt.plot((1/H),LambdaTracker,label='$\Lambda$')
		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel('$1/H$ $1/$GeV')
		plt.ylabel('GeV')
		plt.legend(loc='upper right')


		plt.show()

	return maHRatio, H, nTherm, nBmu, HBmu, Htherm

def InflationCheck(HI,L_i,y):
	Parameters()

	myPhi = phi_i
	myH   = HI

	myFa,myQ,myP	     = faDyn(myH,L_i,y,myPhi)
	maHIRatio	     = (maZT(L_i,myFa,y,myQ,myP)/(3*myH))
	InflationSuppression = (maHIRatio > 1.)

	return InflationSuppression


#Ch 2: Evaluation
#2.0 Write loop to scan over different values of y (begin with just one value y_0 = 10^-6)
#2.1 Write loop to scan over HIxTR meshgrid

def ScanHIxTR(L,y,N):
	GridInit()

	MyHH,MyTT,MymaByHMax = HIxTR(HI_max,HI_min,TR_max,TR_min)

#2.1 Write loop to scan over HIxTR meshgrid
	for n in range(np.shape(MyHH)[0]):
		for m in range(np.shape(MyHH)[1]):
#2.2 For each point, use cosmology functions to find max ma/H 
			MymaHRatioMax, InflBool, HBmuOscBool, HPreThermBool = CosmoEvol(MyHH[n,m],MyTT[n,m],y,L,N)
#2.3 Write each of these points to the output meshgrid during the HIxTR meshgrid scane
			if(MymaHRatioMax >= 1 and (not InflBool)):
				MymaByHMax[n,m] = 2.
			elif(MymaHRatioMax >= 1 and (not HBmuOscBool)):
				MymaByHMax[n,m] = 1.
			elif(MymaHRatioMax <= 1 and (HPreThermBool)):
				MymaByHMax[n,m] = 0.5
			else:
				MymaByHMax[n,m] = MymaHRatioMax
	return MymaByHMax,MyHH,MyTT


def ScanHIxL(y):
	GridInit()

	MyHH,MyLL,MyHHxLLExcl = HIxTR(HI_max,HI_min,L_max,L_min)

	for n in range(np.shape(MyHH)[0]):
		for m in range(np.shape(MyHH)[1]):
			InfBool = InflationCheck(MyHH[n,m],MyLL[n,m],y)
			if(not InfBool):
				MyHHxLLExcl[n,m]=1.
	return MyHHxLLExcl,MyHH,MyLL





#2.4 For new values of Y, we should write additional meshgrids
# We can find the maximum allowed parameter space in a scan over y's in the following way:
	#For each meshgrid in the y scan, we take binary values for ma/H >1, <1 during the critical era, and sum over the entire grid
	#This way, the grid with the most allowed points has the smallest sum, and we plot that one with its y value

#2.5 Binary sum over y meshgrids, minimize the sum
#2.6 Plot using a contour map the minimzed meshgrid

def MHCont(L,y,Npsi):
	ThismaByHMax,ThisHH,ThisTT = ScanHIxTR(L,y,Npsi)

	fig = plt.figure(1)
	myc = plt.contourf(ThisTT,ThisHH,ThismaByHMax,levels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2])
	fig.colorbar(myc)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('$T_R$ GeV')
	plt.ylabel('$H_I$ GeV')



	plt.show()

	return 0

def HLCont(y):
	ThisHILL,ThisHH,ThisLL = ScanHIxL(y)

	fig = plt.figure(1)
	myc = plt.contourf(ThisLL,ThisHH,ThisHILL,levels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1])
	fig.colorbar(myc)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('$\Lambda$ GeV')
	plt.ylabel('$H_I$ GeV')



	plt.show()

	return 0











