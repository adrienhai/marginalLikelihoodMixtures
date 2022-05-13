import sys
import numpy as np
import scipy.stats as ss
from scipy.special import logsumexp
from scipy.special import loggamma
from scipy.optimize import minimize_scalar,minimize

from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import numba as nb
import math

COUNT=0

def readCount():
    global COUNT
    return COUNT

def resetCount():
    global COUNT
    COUNT=0
def catDistLogProb(logProb):
    n=len(logProb)

    #sim n Gumbel RVs
    g=-np.log(-np.log(np.random.random(n)))
    return np.argmax(g+logProb)

@nb.jit(nopython=True)
def _logLikelihoodAllocationsJit(S,N,data,mu0,nu,alpha0,beta0):
    clusters=np.unique(S)
    llk=0
    for k in range(len(clusters)):
        nk=N[clusters[k]]
        yk=data[S==clusters[k]]
        beta_n=beta0+0.5*nk*np.var(yk)+nk*nu/(2*(nu+nk))*(np.mean(yk)-mu0)**2
     
        llk=llk-nk*0.5*np.log(2*np.pi*beta_n)+0.5*np.log(nu/(nu+nk))+alpha0*np.log(beta0/beta_n)+math.lgamma(alpha0+nk/2)-math.lgamma(alpha0)
    return llk  
     
# def counter(func):
#     def wrapper(*args, **kwargs):    # "wrapper" function now exists
#         wrapper.count += 1           # count doesn't exist yet, but will when wrapper is called.
#         return func(*args,**kwargs)  # call the wrapped function and return result
#     wrapper.count = 0                # init the attribute on the function
#     wrapper.__name__  = func.__name__
#     return wrapper


class DPmixnorm1d:
    
    
    def __init__(self, data=None, priorDistLocScale=None,
                 priorParamLocScale=None, priorDistM=None, priorParamM=None):
        if data is None:
            sys.exit('please specify data')
        else:
            self.data=data
        
        
        self.n=data.shape[0]

        if priorDistLocScale is None:
            self.priorDistLocScale='Conjugate'
        else:
            self.priorDistLocScale=priorDistLocScale

        if priorParamLocScale is None:
            if self.priorDistLocScale=='Conjugate':
                self.priorParamLocScale=np.array([np.mean(self.data),2.6/(np.max(self.data)-np.min(self.data)),1.28,0.36*np.var(self.data)]) #b0, N0, c0, C0 according to p.178 Fruhwirth Schnatter, as chsen by raftery 1996
        else:
            self.priorParamLocScale=priorParamLocScale

        if priorDistM is None:
            self.priorDistM='gamma'
        else:
            self.priorDistM=priorDistM

        if priorParamM is None:
            if self.priorDistM=='gamma':
                self.priorParamM=np.array([1,1]) #Shape and SCALE : watch out with def of wiki and scipy
        else:
            self.priorParamM=priorParamM
    
    
    def priorSimM(self):
        if self.priorDistM=='gamma':
            return ss.gamma.rvs(a=self.priorParamM[0],scale=1/self.priorParamM[1]) #Shape and RATE : watch out with def of wiki and scipy
    
    # def logLikelihoodAllocations(self,S,N):
    #     clusters=np.unique(S)
    #     llk=0
    #     for k in range(len(clusters)):
    #         mu0=self.priorParamLocScale[0]
    #         nu=self.priorParamLocScale[1]
    #         alpha0=self.priorParamLocScale[2]
    #         beta0=self.priorParamLocScale[3]
    #         nk=N[clusters[k]]
    #         yk=self.data[S==clusters[k]]
    #         beta_n=beta0+0.5*nk*np.var(yk)+nk*nu/(2*(nu+nk))*(np.mean(yk)-mu0)**2
    #         llk+=-nk*0.5*np.log(2*np.pi*beta_n)+0.5*np.log(nu/(nu+nk))+alpha0*np.log(beta0/beta_n)+loggamma(alpha0+nk/2)-loggamma(alpha0)
    #     return llk
    
    def logLikelihoodAllocations(self,S,N): 
        global COUNT
        COUNT+=1
        return _logLikelihoodAllocationsJit(S=S,N=N,data=self.data,mu0=self.priorParamLocScale[0],nu=self.priorParamLocScale[1],alpha0=self.priorParamLocScale[2],beta0=self.priorParamLocScale[3])
    def logLikelihoodAllocationsTempered(self,S,N,temperature): #just need to change n_k to temperature*n_k
        clusters=np.unique(S)
        llk=0
        for k in range(len(clusters)):
            mu0=self.priorParamLocScale[0]
            nu=self.priorParamLocScale[1]
            alpha0=self.priorParamLocScale[2]
            beta0=self.priorParamLocScale[3]
            #nk=N[clusters[k]]*temperature
            nk=N[clusters[k]]
            yk=self.data[S==clusters[k]]
            beta_n=beta0+0.5*nk*np.var(yk)+nk*nu/(2*(nu+nk))*(np.mean(yk)-mu0)**2
            llk+=-nk*0.5*np.log(2*np.pi*beta_n)+0.5*np.log(nu/(nu+nk))+alpha0*np.log(beta0/beta_n)+loggamma(alpha0+nk/2)-loggamma(alpha0)
        return temperature*llk
    
    def priorAllocationSim(self,M):
        S=np.zeros(self.n, dtype='int')
        N=np.zeros(self.n, dtype='int')
        N[0]=1
        for i in range(1,self.n):
            if M/(M+i)>np.random.random():
                S[i]=np.max(S)+1
                N[S[i]]+=1
            else:
                newIndex=np.random.choice(self.n,p=N/i)
                S[i]=newIndex
                N[newIndex]+=1
        return(S,N)
    def logPrior(self,S,N,M):
        clusters=np.unique(S)
        logPrior=ss.gamma.logpdf(M,a=self.priorParamM[0],scale=1/self.priorParamM[1])
        logPrior+=len(clusters)*np.log(M)+loggamma(M)-loggamma(M+self.n)
        for k in range(len(clusters)):
            logPrior+=loggamma(N[clusters[k]])
        return logPrior
    def SMCAllocResampling(self,numParticles,temperatures,numGibbsMove):
        K=len(temperatures)
        if temperatures[0]!=0 or temperatures[K-1]!=1:
            sys.exit('The first and last elements of the temperatures array must be 0 and 1 respectively')
        S=[]
        N=[]
        M=[]
        Z=np.zeros(K+1)
        Z[0]=0#estimate of the successive ratios of log normalising constants
        #Sample from the prior
        if self.priorDistM!='gamma':
            sys.exit('The choice of prior for the concentration parameter M is not supported yet')
        #print('prior particules')
        for j in range(numParticles):
            M.append(ss.gamma.rvs(a=self.priorParamM[0],scale=self.priorParamM[1]))
            simS,simN=self.priorAllocationSim(M[-1])
            S.append(simS)
            N.append(simN)
            W=np.log(np.ones(numParticles)/numParticles)
            w=np.log(np.ones(numParticles)/numParticles)
            #print(S[j])
        for k in range(1,K):
            
            
            #Resample
            logESS=-logsumexp(2*W)
            print(np.exp(logESS))
            #print('resampling')
            Scopy=S.copy()
            Ncopy=N.copy()
            Mcopy=M.copy()
            for j in range(numParticles):
                #print('Particule before :')
                #print(S[j])
                sampledIndex=catDistLogProb(w)
                S[j]=Scopy[sampledIndex].copy()
                N[j]=Ncopy[sampledIndex].copy()
                M[j]=Mcopy[sampledIndex].copy()
                #print('Particule after :')
                #print(S[j])
            W=np.log(np.ones(numParticles)/numParticles)
            #Mutate
            Scopy=S.copy()
            Ncopy=N.copy()
            Mcopy=M.copy()    
            for j in range(numParticles):
                
                for i in range(numGibbsMove):
                
                    Scopy[j],Ncopy[j],Mcopy[j]=self.MHwithinGibbsAllocMove(Scopy[j],Ncopy[j],Mcopy[j],temperatures[k])
                    
                #print('moved particle')
                #print(S[j],N[j],M[j])
                S[j]=Scopy[j].copy()
                N[j]=Ncopy[j].copy()
                M[j]=Mcopy[j].copy()
            for j in range(numParticles):
                #Reweighting
                w[j]=(temperatures[k]-temperatures[k-1])*self.logLikelihoodAllocations(S[j],N[j])
            Z[k]=logsumexp(w)-np.log(numParticles)
            #Renormalising the log weights
            W=w-(Z[k]+np.log(numParticles))
            #W=w-Z[k]
        #print(S)
        #print(N)
        #print(M)
        print(np.sum(Z))
        return(S,N,M,np.sum(Z))

    def SMCAlloc(self,numParticles,temperatures,numGibbsMove):
        K=len(temperatures)
        if temperatures[0]!=0 or temperatures[K-1]!=1:
            sys.exit('The first and last elements of the temperatures array must be 0 and 1 respectively')
        S=[]
        N=[]
        M=[]
        Z=np.zeros(K) #estimate of the successive ratios of normalising constants
        #Sample from the prior
        if self.priorDistM!='gamma':
            sys.exit('The choice of prior for the concentration parameter M is not supported yet')
        #print('prior particules')
        for j in range(numParticles):
            M.append(ss.gamma.rvs(a=self.priorParamM[0],scale=self.priorParamM[1]))
            simS,simN=self.priorAllocationSim(M[-1])
            S.append(simS)
            N.append(simN)
            W=np.log(np.ones(numParticles)/numParticles)
            w=np.log(np.ones(numParticles)/numParticles)
            #print(S[j])
        for k in range(1,K):
            for j in range(numParticles):
                #Reweighting
                w[j]=W[j]+(temperatures[k]-temperatures[k-1])*self.logLikelihoodAllocations(S[j],N[j])
            Z[k]=logsumexp(w)
            #Renormalising the log weights
            W=w-Z[k]
            
            #Resample
            logESS=-logsumexp(2*W)
            print(np.exp(logESS))
            if logESS<np.log(0.8*numParticles):#Resample multinomially
                #print('resampling')
                Scopy=S.copy()
                Ncopy=N.copy()
                Mcopy=M.copy()
                for j in range(numParticles):
                    #print('Particule before :')
                    #print(S[j])
                    sampledIndex=catDistLogProb(w)
                    S[j]=Scopy[sampledIndex].copy()
                    N[j]=Ncopy[sampledIndex].copy()
                    M[j]=Mcopy[sampledIndex].copy()
                    #print('Particule after :')
                    #print(S[j])
                W=np.log(np.ones(numParticles)/numParticles)
            #Mutate
            Scopy=S.copy()
            Ncopy=N.copy()
            Mcopy=M.copy()    
            for j in range(numParticles):
                
                for i in range(numGibbsMove):
                
                    Scopy[j],Ncopy[j],Mcopy[j]=self.MHwithinGibbsAllocMove(Scopy[j],Ncopy[j],Mcopy[j],temperatures[k])
                    
                #print('moved particle')
                #print(S[j],N[j],M[j])
                S[j]=Scopy[j].copy()
                N[j]=Ncopy[j].copy()
                M[j]=Mcopy[j].copy()
        #print(S)
        #print(N)
        #print(M)
        print(np.sum(Z))
        return(S,N,M,np.sum(Z))
    
    def adaptiveSMCAlloc2(self,numParticles,numGibbsMove,ESSThreshold,maxIterBissection=10000,TOL=0.000000005):
        temperatures=[]
        temperatures.append(0)

        
        S=[]
        N=[]
        M=[]
        Z=[]
        Z.append(0)#estimate of the successive ratios of normalising constants
        #Sample from the prior
        if self.priorDistM!='gamma':
            sys.exit('The choice of prior for the concentration parameter M is not supported yet')
        #print('prior particules')
        for j in range(numParticles):
            M.append(ss.gamma.rvs(a=self.priorParamM[0],scale=self.priorParamM[1]))
            simS,simN=self.priorAllocationSim(M[-1])
            S.append(simS)
            N.append(simN)
            W=np.log(np.ones(numParticles)/numParticles)
            w=np.log(np.ones(numParticles)/numParticles)
        
        k=0
        while temperatures[k]<1:
            #Find the next temperature adaptively
            llkParticles=np.zeros(numParticles)
            for j in range(numParticles):
                    llkParticles[j]=self.logLikelihoodAllocations(S[j],N[j])
            #Try temperature = 1
            wtemp=np.zeros(numParticles)
            for j in range(numParticles):
                wtemp[j]=(1-temperatures[k])*llkParticles[j]
            Wtemp=wtemp-logsumexp(wtemp)
            logESS=-logsumexp(2*Wtemp)
            if logESS>np.log(ESSThreshold*numParticles):
                temperatures.append(1)
            #else do bissection algorithm
            else:
                #print('statr')
                a=temperatures[k]
                b=1
                l=0
                while l<maxIterBissection:
                    tempcand=(a+b)/2
                    #print(tempcand)
                    for j in range(numParticles):
                        wtemp[j]=(tempcand-temperatures[k])*llkParticles[j]
                    Wtemp=wtemp-logsumexp(wtemp)
                    logESS=-logsumexp(2*Wtemp)
                    if np.abs(logESS-np.log(ESSThreshold*numParticles))<np.log(TOL) or ((b-a)/2)<TOL:
                        break
                    else:
                        if logESS>np.log(ESSThreshold*numParticles): #need to increase the temp
                            a=tempcand
                        else:
                            b=tempcand
                    l=l+1
                    
                temperatures.append(tempcand)
                #print(l)
            print(temperatures[k],np.exp(logESS))
            k=k+1
            for j in range(numParticles):
                #Reweighting
                w[j]=W[j]+(temperatures[k]-temperatures[k-1])*llkParticles[j]
            Z.append(logsumexp(w))
            #Renormalising the log weights
            W=w-Z[k]
            #Resample
            logESS=-logsumexp(2*W)
            #print(np.exp(logESS))
            if logESS<np.log(0.8*numParticles):#Resample multinomially
                #print('resampling')
                Scopy=S.copy()
                Ncopy=N.copy()
                Mcopy=M.copy()
                for j in range(numParticles):
                    #print('Particule before :')
                    #print(S[j])
                    sampledIndex=catDistLogProb(w)
                    S[j]=Scopy[sampledIndex].copy()
                    N[j]=Ncopy[sampledIndex].copy()
                    M[j]=Mcopy[sampledIndex].copy()
                    #print('Particule after :')
                    #print(S[j])
                W=np.log(np.ones(numParticles)/numParticles)
            #Mutate
            Scopy=S.copy()
            Ncopy=N.copy()
            Mcopy=M.copy()    
            for j in range(numParticles):
                
                for i in range(numGibbsMove):
                
                    Scopy[j],Ncopy[j],Mcopy[j]=self.MHwithinGibbsAllocMove(Scopy[j],Ncopy[j],Mcopy[j],temperatures[k])
                    
                #print('moved particle')
                #print(S[j],N[j],M[j])
                S[j]=Scopy[j].copy()
                N[j]=Ncopy[j].copy()
                M[j]=Mcopy[j].copy()
            
        #print(S)
        #print(N)
        #print(M)
        print(np.sum(Z))
        return(np.sum(Z))
        
    def adaptiveSMCAlloc(self,MChopin,PChopin,ESSThreshold,maxIterBissection=10000,TOL=0.000000005):
        temperatures=[]
        temperatures.append(0)
        numParticles=MChopin*PChopin

        
        S=[]
        N=[]
        M=[]
        Z=[]
        Z.append(0)#estimate of the successive ratios of normalising constants
        #Sample from the prior
        if self.priorDistM!='gamma':
            sys.exit('The choice of prior for the concentration parameter M is not supported yet')
        #print('prior particules')
        for j in range(numParticles):
            M.append(ss.gamma.rvs(a=self.priorParamM[0],scale=self.priorParamM[1]))
            simS,simN=self.priorAllocationSim(M[-1])
            S.append(simS)
            N.append(simN)
            W=np.log(np.ones(numParticles)/numParticles)
            w=np.log(np.ones(numParticles)/numParticles)
            
        k=0
        while temperatures[k]<1:
            #Find the next temperature adaptively
            llkParticles=np.zeros(numParticles)
            for j in range(numParticles):
                    llkParticles[j]=self.logLikelihoodAllocations(S[j],N[j])
            #Try temperature = 1
            wtemp=np.zeros(numParticles)
            for j in range(numParticles):
                wtemp[j]=(1-temperatures[k])*llkParticles[j]
            Wtemp=wtemp-logsumexp(wtemp)
            logESS=-logsumexp(2*Wtemp)
            if logESS>np.log(ESSThreshold*numParticles):
                temperatures.append(1)
            #else do bissection algorithm
            else:
                print('statr')
                a=temperatures[k]
                b=1
                l=0
                while l<maxIterBissection:
                    tempcand=(a+b)/2
                    print(tempcand)
                    for j in range(numParticles):
                        wtemp[j]=(tempcand-temperatures[k])*llkParticles[j]
                    Wtemp=wtemp-logsumexp(wtemp)
                    logESS=-logsumexp(2*Wtemp)
                    if np.abs(logESS-np.log(ESSThreshold*numParticles))<np.log(TOL) or ((b-a)/2)<TOL:
                        break
                    else:
                        if logESS>np.log(ESSThreshold*numParticles): #need to increase the temp
                            a=tempcand
                        else:
                            b=tempcand
                    l=l+1
                    
                temperatures.append(tempcand)
                print(l)
            print(temperatures[k],np.exp(logESS))
            Scopy=S.copy()
            Ncopy=N.copy()
            Mcopy=M.copy() 
            #Resample and move
            for j in range(MChopin):
                #print('Particule before :')
                #print(S[j])
                sampledIndex=catDistLogProb(w)
                Scopy[j*PChopin]=S[sampledIndex].copy()
                Ncopy[j*PChopin]=N[sampledIndex].copy()
                Mcopy[j*PChopin]=M[sampledIndex].copy()
                for p in range(1,PChopin):
                    Scopy[j*PChopin+p],Ncopy[j*PChopin+p],Mcopy[j*PChopin+p]=self.MHwithinGibbsAllocMove(Scopy[j*PChopin+p-1],Ncopy[j*PChopin+p-1],Mcopy[j*PChopin+p-1],temperatures[k+1])
            S=Scopy.copy()
            N=Ncopy.copy()
            M=Mcopy.copy()
            
            #Reweight
            for j in range(numParticles):
                #Reweighting
                w[j]=(temperatures[k+1]-temperatures[k])*self.logLikelihoodAllocations(S[j],N[j])
            Z.append(-np.log(numParticles)+logsumexp(w))
            #Renormalising the log weights
            W=w-(Z[k+1]+np.log(numParticles))
               
            
            
            k=k+1
        return(np.sum(Z))
    def wasteFreeSMCAlloc(self,MChopin,PChopin,temperatures):  ###resultats étranges, à corriger
        numParticles=MChopin*PChopin
        K=len(temperatures)
        if temperatures[0]!=0 or temperatures[K-1]!=1:
            sys.exit('The first and last elements of the temperatures array must be 0 and 1 respectively')
        S=[]
        N=[]
        M=[]
        Z=np.zeros(K) #estimate of the successive ratios of normalising constants
        #Sample from the prior
        if self.priorDistM!='gamma':
            sys.exit('The choice of prior for the concentration parameter M is not supported yet')
        #print('prior particules')
        for j in range(numParticles):
            M.append(ss.gamma.rvs(a=self.priorParamM[0],scale=self.priorParamM[1]))
            simS,simN=self.priorAllocationSim(M[-1])
            S.append(simS)
            N.append(simN)
            W=np.log(np.ones(numParticles)/numParticles)
            w=np.log(np.ones(numParticles)/numParticles)
            #print(S[j])
        for k in range(1,K):
            
            Scopy=S.copy()
            Ncopy=N.copy()
            Mcopy=M.copy() 
            #Resample and move
            for j in range(MChopin):
                #print('Particule before :')
                #print(S[j])
                sampledIndex=catDistLogProb(w)
                Scopy[j*PChopin]=S[sampledIndex].copy()
                Ncopy[j*PChopin]=N[sampledIndex].copy()
                Mcopy[j*PChopin]=M[sampledIndex].copy()
                for p in range(1,PChopin):
                    Scopy[j*PChopin+p],Ncopy[j*PChopin+p],Mcopy[j*PChopin+p]=self.MHwithinGibbsAllocMove(Scopy[j*PChopin+p-1],Ncopy[j*PChopin+p-1],Mcopy[j*PChopin+p-1],temperatures[k])
            S=Scopy.copy()
            N=Ncopy.copy()
            M=Mcopy.copy()
            
            #Reweight
            for j in range(numParticles):
                #Reweighting
                w[j]=(temperatures[k]-temperatures[k-1])*self.logLikelihoodAllocations(S[j],N[j])
            Z[k]=-np.log(numParticles)+logsumexp(w)
            #Renormalising the log weights
            W=w-(Z[k]+np.log(numParticles))
        print(np.sum(Z))   
        return 0
    
    def GibbsSamplingClustersSplitAndMerge(self,numIter,numIterLaunch): #Jain and Neal (2004)
        S=[]
        N=[]
        M=[]
        eta=[] #augmentation variable to compute posterior full conditional on M
        K=np.zeros(numIter,dtype=int)
        
        #intialise from the prior
        S.append(np.zeros(self.n,dtype=int))
        N.append(np.zeros(self.n,dtype=int))
        M.append(self.priorSimM())
        S[0],N[0]=self.priorAllocationSim(M[0])
        clusters=np.unique(S[0])
        K[0]=len(clusters)
        
        for t in range(1,numIter):
            S.append(np.zeros(self.n,dtype=int))
            S[t]=S[t-1].copy()
            N.append(np.zeros(self.n,dtype=int))
            N[t]=N[t-1].copy()
            M.append(0)
            M[t]=M[t-1].copy()
            K[t]=K[t-1].copy()
            splittedIndicator=0
            print(S[t])
            #Sample two indices at random for split or merge
            idx1,idx2=np.random.choice(self.n,size=2,replace=False)
            print(idx1,idx2)
            #activeSet=S in the original article
            activeSet=np.array(np.where(np.logical_or(S[t]==S[t][idx1],S[t]==S[t][idx2])))
            print(activeSet)
            activeSet=activeSet[(activeSet!=idx1) & (activeSet!=idx2)]
            print(activeSet)
            is_empty=(activeSet.size==0)
            #If the activeSet is empty
            
            if is_empty==True:
            
                
                #Coompute the likelihood ratio for the MH acceptance probabibility
                ytemp=np.array([self.data[idx1]])
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                llk_split=DPmixtemp.logLikelihoodAllocations(Stemp, Ntemp)
                
                ytemp=np.array([self.data[idx2]])
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                llk_split+=DPmixtemp.logLikelihoodAllocations(Stemp, Ntemp)
                
                ytemp=self.data[[idx1,idx2]]
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                Ntemp=np.array([DPmixtemp.n]).copy()
                llk_merge=DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                
                if S[t][idx1]==S[t][idx2]:
                    #Propose to split into two components
                    print(llk_split-llk_merge+np.log(M[t]))
                    if np.log(np.random.random())<llk_split-llk_merge+np.log(M[t]):
                        indexClusterAvailable=next((idx for idx, val in np.ndenumerate(N[t]) if val==0))[0] #Find the first zero cluster
                        N[t][indexClusterAvailable]+=1
                        N[t][S[t][idx1]]-=1
                        #clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                        S[t][idx1]=indexClusterAvailable
                        K[t]+=1
                        print('split accepted')
                else:
                    #Propose to merge into a single component
                    print(llk_merge-llk_split-np.log(M[t]))
                    if np.log(np.random.random())<llk_merge-llk_split-np.log(M[t]):
                        N[t][S[t][idx1]]-=1
                        #clusters=np.setdiff1d(clusters,np.array([S[t][idx1]]))
                        S[t][idx1]=S[t][idx2]
                        N[t][S[t][idx2]]+=1
                        K[t]=K[t]-1
                        print('merge accepted')
                print(S[t],N[t],K[t])
            else : #The active set is not empty
                Slaunch=S[t].copy()
                Nlaunch=N[t].copy()
                Klaunch=K[t]
                #clustersLaunch=clusters
                if S[t][idx1]==S[t][idx2]:
                    indexClusterAvailable=next((idx for idx, val in np.ndenumerate(Nlaunch) if val==0))[0]
                    Nlaunch[Slaunch[idx1]]-=1
                    Nlaunch[indexClusterAvailable]+=1
                    #clustersLaunch=np.insert(clustersLaunch,indexClusterAvailable,indexClusterAvailable)
                    Slaunch[idx1]=indexClusterAvailable
                    Klaunch+=1
                #else if the alloc of idx1 and idx2 are different then do nothing
                for k in activeSet:
                    #assigne the other obs to either the clusters of idx1 or idx2 with prob 1/2
                    if np.random.random()<0.5:
                        Nlaunch[Slaunch[k]]-=1
                        Slaunch[k]=Slaunch[idx1]
                        Nlaunch[Slaunch[idx1]]+=1
                    else :
                        Nlaunch[Slaunch[k]]-=1
                        Slaunch[k]=Slaunch[idx2]
                        Nlaunch[Slaunch[idx2]]+=1
                for l in range(numIterLaunch):
                    for k in activeSet:
                        Nlaunch[Slaunch[k]]-=1 #remove obs k from its previous allocation
                        
                        
                        logProbs=np.zeros(2)
                        
                        Slaunch[k]=Slaunch[idx1] #temporarily pretend obs k is in cluster of idx1
                        ytemp=self.data[Slaunch==Slaunch[idx1]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[0]=np.log(Nlaunch[Slaunch[idx1]])+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        Slaunch[k]=Slaunch[idx2] #temporarily pretend obs k is in cluster of idx2
                        ytemp=self.data[Slaunch==Slaunch[idx2]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[1]=np.log(Nlaunch[Slaunch[idx2]])+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        sampledCluster=catDistLogProb(logProbs)
                        
                        if sampledCluster==0: #assigned to the cluster of idx1
                            Nlaunch[Slaunch[idx1]]+=1
                            Slaunch[k]=Slaunch[idx1]
                            
                        else : #assigned to the cluster of idx2
                            Nlaunch[Slaunch[idx2]]+=1
                            Slaunch[k]=Slaunch[idx2]
                print(Slaunch,Nlaunch,Klaunch)
                if S[t][idx1]==S[t][idx2]: #propose a split move
                    Ssplit=Slaunch.copy()
                    Nsplit=Nlaunch.copy()
                    Ksplit=Klaunch
                    
                    q=0 #product for the MH acceptance ratio
                    for k in activeSet:
                        
                        Nsplit[Ssplit[k]]-=1 #remove obs k from its previous allocation
                        
                        
                        logProbs=np.zeros(2)
                        
                        Ssplit[k]=Ssplit[idx1] #temporarily pretend obs k is in cluster of idx1
                        ytemp=self.data[Ssplit==Ssplit[idx1]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[0]=np.log(Nsplit[Ssplit[idx1]])+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        Ssplit[k]=Ssplit[idx2] #temporarily pretend obs k is in cluster of idx2
                        ytemp=self.data[Ssplit==Ssplit[idx2]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[1]=np.log(Nsplit[Ssplit[idx2]])+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        sampledCluster=catDistLogProb(logProbs)
                        
                        q+=logProbs[sampledCluster]-logsumexp(logProbs)
                        
                        if sampledCluster==0: #assigned to the cluster of idx1
                            Nsplit[Ssplit[idx1]]+=1
                            Ssplit[k]=Ssplit[idx1]
                            
                        else : #assigned to the cluster of idx2
                            Nsplit[Ssplit[idx2]]+=1
                            Ssplit[k]=Ssplit[idx2]
                            
                    #Coompute the likelihood ratio for the MH acceptance probabibility
                    ytemp=self.data[Ssplit==Ssplit[idx1]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    llk_split=DPmixtemp.logLikelihoodAllocations(Stemp, Ntemp)
                
                    ytemp=self.data[Ssplit==Ssplit[idx2]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    llk_split+=DPmixtemp.logLikelihoodAllocations(Stemp, Ntemp)
                    
                    ytemp=self.data[S[t]==S[t][idx1]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    llk_merge=DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    if np.log(np.random.random())<(llk_split-llk_merge+np.log(M[t])+np.log(np.math.factorial(Nsplit[Ssplit[idx1]]-1)*np.math.factorial(Nsplit[Ssplit[idx2]]-1)/np.math.factorial(N[t][S[t][idx1]]-1))-q):
                        #print('llk',llk_split-llk_merge,'alpha',np.log(M[t]),'prior',np.log(np.math.factorial(Nsplit[Ssplit[idx1]]-1)*np.math.factorial(Nsplit[Ssplit[idx2]]-1)/np.math.factorial(N[t][S[t][idx1]]-1)),'proposal',q)
                        S[t]=Ssplit.copy()
                        N[t]=Nsplit.copy()
                        K[t]=Ksplit.copy()
                        splitted=1
                        print('split accepted')
                    else:
                        #print('llk',llk_split-llk_merge,'alpha',np.log(M[t]),'prior',np.log(np.math.factorial(Nsplit[Ssplit[idx1]]-1)*np.math.factorial(Nsplit[Ssplit[idx2]]-1)/np.math.factorial(N[t][S[t][idx1]]-1)),'proposal',q)

                        print(llk_split-llk_merge+np.log(M[t])+np.log(np.math.factorial(Nsplit[Ssplit[idx1]]-1)*np.math.factorial(Nsplit[Ssplit[idx2]]-1)/np.math.factorial(N[t][S[t][idx1]]-1))-q)
                        splitted=0
                        print('split rejected')
                
                else : #propose a merge move
                    Smerge=S[t].copy()
                    Nmerge=N[t].copy()
                    Kmerge=K[t]-1 #whatever happens, exactly one cluster will disappear
                    
                    Nmerge[S[t][idx1]]-=1
                    Smerge[idx1]=S[t][idx2]
                    Nmerge[S[t][idx2]]+=1
                    q=0 #will be the sum of log probs of going from the launch state to the original allocation s_k
                    for k in activeSet:
                        Nmerge[Smerge[k]]-=1
                        Smerge[k]=S[t][idx2]
                        Nmerge[Smerge[k]]+=1
                        
                        Nlaunch[Slaunch[k]]-=1
                        Slaunch[k]=S[t][idx1] #pretend obs k launch is in cluster of idx1
                        #Nlaunch[S[t][idx1]]+=1
                        
                        logProbs=np.zeros(2)
                        ytemp=self.data[Slaunch==S[t][idx1]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[0]=np.log(Nlaunch[S[t][idx1]])+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        
                        #Nlaunch[Slaunch[k]]-=1
                        Slaunch[k]=S[t][idx2] #pretend obs k launch is in cluster of idx2
                        #Nlaunch[S[t][idx2]]+=1
                        
        
                        ytemp=self.data[Slaunch==S[t][idx2]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[1]=np.log(Nlaunch[S[t][idx2]])+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        if S[t][k]==S[t][idx1]:
                            Slaunch[k]=S[t][idx1]
                            Nlaunch[S[t][idx1]]+=1
                            q+=logProbs[0]-logsumexp(logProbs)
                        elif S[t][k]==S[t][idx2]:
                            Slaunch[k]=S[t][idx2]
                            Nlaunch[S[t][idx2]]+=1
                            q+=logProbs[1]-logsumexp(logProbs)
                        else:
                            sys.exit('error : active set ill-definied')
                    
                    #Coompute the likelihood ratio for the MH acceptance probabibility
                    ytemp=self.data[S[t]==S[t][idx1]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    llk_split=DPmixtemp.logLikelihoodAllocations(Stemp, Ntemp)
                
                    ytemp=self.data[S[t]==S[t][idx2]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    llk_split+=DPmixtemp.logLikelihoodAllocations(Stemp, Ntemp)
                    
                    ytemp=self.data[Smerge==S[t][idx2]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    llk_merge=DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    if np.log(np.random.random())<(-np.log(M[t])+np.log((np.math.factorial(Nmerge[S[t][idx2]]-1))/(np.math.factorial(N[t][S[t][idx1]]-1)*np.math.factorial(N[t][S[t][idx2]]-1)))+llk_merge-llk_split+q):
                        #print('alpha',-np.log(M[t]),'prior',np.log((np.math.factorial(Nmerge[S[t][idx2]]-1))/(np.math.factorial(N[t][S[t][idx1]]-1)*np.math.factorial(N[t][S[t][idx2]]-1))),'llk',llk_merge-llk_split,'proposal',q)
                        S[t]=Smerge.copy()
                        N[t]=Nmerge.copy()
                        K[t]=Kmerge.copy()
                        print('merge accepted')
                    else:
                        #print('alpha',-np.log(M[t]),'prior',np.log((np.math.factorial(Nmerge[S[t][idx2]]-1))/(np.math.factorial(N[t][S[t][idx1]]-1)*np.math.factorial(N[t][S[t][idx2]]-1))),'llk',llk_merge-llk_split,'proposal',q)
                        print('merge rejected')
            #Sample concentration parameter M
            ##Sample eta, latent var
            #print('K',K)
            eta.append(ss.beta.rvs(a=M[t]+1,b=self.n))
            epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta[-1]))+self.priorParamM[0]+K[t]-1)
            #epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta)))
            if(np.random.random()<epsilon):
                M[t]=ss.gamma.rvs(a=self.priorParamM[0]+K[t], scale=1/(self.priorParamM[1]-np.log(eta[-1])))
            else:
                M[t]=ss.gamma.rvs(a=self.priorParamM[0]+K[t]-1, scale=1/(self.priorParamM[1]-np.log(eta[-1])))
        return S,N,M,K,eta
    def GibbsSamplingClusters(self,numIter):
        S=[]
        N=[]
        M=[]
        eta=[] #augmentation variable to compute posterior full conditional on M
        K=np.zeros(numIter,dtype=int)
        
        #intialise from the prior
        S.append(np.zeros(self.n,dtype=int))
        N.append(np.zeros(self.n,dtype=int))
        M.append(self.priorSimM())
        S[0],N[0]=self.priorAllocationSim(M[0])
        clusters=np.unique(S[0])
        K[0]=len(clusters)
        
        for t in range(1,numIter):
            #print(t/numIter*100,end="\r")
            S.append(np.zeros(self.n,dtype=int))
            S[t]=S[t-1].copy()
            N.append(np.zeros(self.n,dtype=int))
            N[t]=N[t-1].copy()
            M.append(0)
            M[t]=M[t-1].copy()
            K[t]=K[t-1].copy()
            
            for i in range(self.n):
                
                #Remove y_i from the data
                NminusI=N[t].copy()
                NminusI[S[t][i]]=NminusI[S[t][i]]-1
                if NminusI[S[t][i]]==0:
                    #A cluster is being emptied
                    K[t]=K[t]-1
                    clusters=np.setdiff1d(clusters,np.array([S[t][i]]))
                logProbs=np.zeros(K[t]+1)
                for k in range(K[t]):
                    
                    logProbs[k]=logProbs[k]+np.log(NminusI[clusters[k]]/(self.n-1+M[t]))
                    Stemp=S[t].copy()
                    Stemp[i]=clusters[k]#pretend y is in cluster k
                    ytemp=self.data[Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[t].copy()
                    Stemp[i]=self.n+1
                    ytemp=self.data[Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int)
                    Ntemp=np.array([DPmixtemp.n])
                    logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                 
                    
                
                logProbs[K[t]]=logProbs[K[t]]+np.log(M[t]/(self.n-1+M[t]))
                ytemp=np.array([self.data[i]])
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                
                logProbs[K[t]]=logProbs[K[t]]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)

                sampledCluster=catDistLogProb(logProbs)
                
                if sampledCluster==K[t]:
                    indexClusterAvailable=next((idx for idx, val in np.ndenumerate(NminusI) if val==0))[0] #Find the first zero cluster
                    NminusI[indexClusterAvailable]+=1
                    clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                    S[t][i]=indexClusterAvailable
                    N[t]=NminusI.copy()
                    K[t]+=1
                    #print(K[t],indexClusterAvailable,N[t],S[t])
                else:
                    #print('assigned to existing cluster',clusters[sampledCluster])
                    S[t][i]=clusters[sampledCluster]
                    NminusI[clusters[sampledCluster]]+=1
                    N[t]=NminusI.copy()
                

            #Sample concentration parameter M
            ##Sample eta, latent var
            #print('K',K)
            eta.append(ss.beta.rvs(a=M[t]+1,b=self.n))
            epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta[-1]))+self.priorParamM[0]+K[t]-1)
            #epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta)))
            if(np.random.random()<epsilon):
                M[t]=ss.gamma.rvs(a=self.priorParamM[0]+K[t], scale=1/(self.priorParamM[1]-np.log(eta[-1])))
            else:
                M[t]=ss.gamma.rvs(a=self.priorParamM[0]+K[t]-1, scale=1/(self.priorParamM[1]-np.log(eta[-1])))
        
        return(S,N,M,K,eta)
    
    def ChibEstimator(self,numIterGibbs,burnIn):
        
        
        #posterior estimation
        S,N,M,K=self.GibbsSamplingClusters(numIterGibbs)
        MStar=np.median(M[burnIn:])
        posteriorM=0
        
        for t in range(burnIn,numIterGibbs):
            eta=ss.beta.rvs(a=MStar+1,b=self.n)
            epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta))+self.priorParamM[0]+K[t]-1)
            posteriorM+=epsilon*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t], scale=1/(self.priorParamM[1]-np.log(eta)))+(1-epsilon)*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t]-1, scale=1/(self.priorParamM[1]-np.log(eta)))
            #print(epsilon)
            
        logposteriorM=-np.log(numIterGibbs-burnIn)+np.log(posteriorM)
            
        #Likelihood estimation
        w=np.zeros(numIterGibbs-burnIn)
        DPmixtemp=DPmixnorm1d(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
        Stemp=np.array([0])
        Ntemp=np.array([1])
        w+=DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
        for t in range(burnIn,numIterGibbs):
            for i in range(1,self.n):
                Stemp=S[t][:i].copy()
                
                clusters=np.unique(Stemp)
                K=len(clusters)
                
                u=np.zeros(K+1)
                for k in range(K):
                    n_k=np.sum(Stemp==clusters[k])
                    ytemp_k=self.data[:i][Stemp==clusters[k]]
                    #print(ytemp_k)
                    ytemp_k_with_i=np.concatenate([ytemp_k,np.array([self.data[i]])])
                    DPmixtemp_k=DPmixnorm1d(data=ytemp_k,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    DPmixtemp_k_with_i=DPmixnorm1d(data=ytemp_k_with_i,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)

                    u[k]=np.log(n_k/(MStar+i))+DPmixtemp_k_with_i.logLikelihoodAllocations(np.zeros(DPmixtemp_k_with_i.n,dtype=int),np.array([DPmixtemp_k_with_i.n]))-DPmixtemp_k.logLikelihoodAllocations(np.zeros(DPmixtemp_k.n,dtype=int),np.array([DPmixtemp_k.n]))
                DPmixtemp=DPmixnorm1d(data=self.data[i],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                #print(DPmixtemp.n)
                u[K]=np.log(MStar/(MStar+i))+DPmixtemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
                w[t-burnIn]+=logsumexp(u)
        llk=-np.log(numIterGibbs-burnIn)+logsumexp(w)
        logprior=ss.gamma.logpdf(MStar,a=self.priorParamM[0],scale=1/self.priorParamM[1])
        print('log posterior', logposteriorM)
        print('llk', llk)
        print('logprior',logprior)
        return llk+logprior-logposteriorM
    
    def ChibEstimator2(self,numIterGibbs,burnIn):
        
        
        #posterior estimation
        S,N,M,K,eta=self.GibbsSamplingClusters(numIterGibbs)
        MStar=np.median(M[burnIn:])
        posteriorM=0
        for t in range(burnIn,numIterGibbs):
            #eta=ss.beta.rvs(a=MStar+1,b=self.n)
            epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta[t-1]))+self.priorParamM[0]+K[t]-1)
            #epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta)))
            posteriorM+=(epsilon*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t], scale=1/(self.priorParamM[1]-np.log(eta[t-1])))+(1-epsilon)*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t]-1, scale=1/(self.priorParamM[1]-np.log(eta[t-1]))))
            #print(epsilon)
            
        logposteriorM=-np.log(numIterGibbs-burnIn)+np.log(posteriorM)
            
        #Likelihood estimation
        w=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            u=np.zeros(self.n)
            S=np.zeros(self.n,dtype=int)
            N=np.zeros(self.n,dtype=int)
            DPmixTemp=DPmixnorm1d(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
            N[0]+=1
            clusters=np.array([0])
            K=1
            u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
            for i in range(1,self.n):
                logProbs=np.zeros(K+1)
                for k in range(K):
                    logProbs[k]+=np.log(N[clusters[k]]/(i+MStar))
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=clusters[k]#pretend y is in cluster k
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=self.n+1
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int)
                    Ntemp=np.array([DPmixtemp.n])
                    logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                logProbs[K]=logProbs[K]+np.log(MStar/(i+MStar))
                ytemp=np.array([self.data[i]])
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)

                sampledCluster=catDistLogProb(logProbs)
                
                if sampledCluster==K:
                    indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                    N[indexClusterAvailable]+=1
                    clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                    S[i]=indexClusterAvailable
                    K+=1
                    #print(K[t],indexClusterAvailable,N[t],S[t])
                else:
                    #print('assigned to existing cluster',clusters[sampledCluster])
                    S[i]=clusters[sampledCluster]
                    N[clusters[sampledCluster]]+=1
                #print(i,clusters,K)
                
                u[i]=logsumexp(logProbs)
            w[t]=np.sum(u)
        llk=-np.log(numIterGibbs-burnIn)+logsumexp(w)
        logprior=ss.gamma.logpdf(MStar,a=self.priorParamM[0],scale=1/self.priorParamM[1])
        #print('llk uncertainty sd', 1/np.sqrt(numIterGibbs-burnIn)*np.sqrt(np.var(np.exp(w)))/np.mean(np.exp(w)))
        #print('log posterior', logposteriorM)
        #print('llk', llk)
        #print('logprior',logprior)
        return llk+logprior-logposteriorM
    
    def ChibEstimator2v2(self,numIterGibbs,burnIn,numIterSIS): #possibility to choose different number of SIS and MCMC simulations
        
        
        #posterior estimation
        S,N,M,K,eta=self.GibbsSamplingClusters(numIterGibbs)
        MStar=np.median(M[burnIn:])
        posteriorM=0
        for t in range(burnIn,numIterGibbs):
            #eta=ss.beta.rvs(a=MStar+1,b=self.n)
            epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta[t-1]))+self.priorParamM[0]+K[t]-1)
            #epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta)))
            posteriorM+=(epsilon*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t], scale=1/(self.priorParamM[1]-np.log(eta[t-1])))+(1-epsilon)*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t]-1, scale=1/(self.priorParamM[1]-np.log(eta[t-1]))))
            #print(epsilon)
            
        logposteriorM=-np.log(numIterGibbs-burnIn)+np.log(posteriorM)
            
        #Likelihood estimation
        w=np.zeros(numIterSIS)
        for t in range(numIterSIS):
            u=np.zeros(self.n)
            S=np.zeros(self.n,dtype=int)
            N=np.zeros(self.n,dtype=int)
            DPmixTemp=DPmixnorm1d(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
            N[0]+=1
            clusters=np.array([0])
            K=1
            u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
            for i in range(1,self.n):
                logProbs=np.zeros(K+1)
                for k in range(K):
                    logProbs[k]+=np.log(N[clusters[k]]/(i+MStar))
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=clusters[k]#pretend y is in cluster k
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=self.n+1
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int)
                    Ntemp=np.array([DPmixtemp.n])
                    logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                logProbs[K]=logProbs[K]+np.log(MStar/(i+MStar))
                ytemp=np.array([self.data[i]])
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)

                sampledCluster=catDistLogProb(logProbs)
                
                if sampledCluster==K:
                    indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                    N[indexClusterAvailable]+=1
                    clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                    S[i]=indexClusterAvailable
                    K+=1
                    #print(K[t],indexClusterAvailable,N[t],S[t])
                else:
                    #print('assigned to existing cluster',clusters[sampledCluster])
                    S[i]=clusters[sampledCluster]
                    N[clusters[sampledCluster]]+=1
                #print(i,clusters,K)
                
                u[i]=logsumexp(logProbs)
            w[t]=np.sum(u)
        llk=-np.log(numIterSIS)+logsumexp(w)
        logprior=ss.gamma.logpdf(MStar,a=self.priorParamM[0],scale=1/self.priorParamM[1])
        print('llk uncertainty sd', 1/np.sqrt(numIterSIS)*np.sqrt(np.var(np.exp(w)))/np.mean(np.exp(w)))
        print('log posterior', logposteriorM)
        print('llk', llk)
        print('logprior',logprior)
        return llk+logprior-logposteriorM
    def ChibEstimator3(self,numIterGibbs,burnIn): #introducing data randomly in the likelihood ordinate estimator
        
        
        #posterior estimation
        S,N,M,K,eta=self.GibbsSamplingClusters(numIterGibbs)
        MStar=np.median(M[burnIn:])
        posteriorM=0
        posterior=np.zeros(numIterGibbs-burnIn)
        for t in range(burnIn,numIterGibbs):
            #eta=ss.beta.rvs(a=MStar+1,b=self.n)
            epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta[t-1]))+self.priorParamM[0]+K[t]-1)
            #epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta)))
            posterior[t-burnIn]=(epsilon*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t], scale=1/(self.priorParamM[1]-np.log(eta[t-1])))+(1-epsilon)*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t]-1, scale=1/(self.priorParamM[1]-np.log(eta[t-1]))))
            posteriorM+=(epsilon*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t], scale=1/(self.priorParamM[1]-np.log(eta[t-1])))+(1-epsilon)*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t]-1, scale=1/(self.priorParamM[1]-np.log(eta[t-1]))))
            #print(epsilon)
            
        logposteriorM=-np.log(numIterGibbs-burnIn)+np.log(posteriorM)
        #Compute variance of posterior ordinate using Chibb 1995
        q=100
        Omega=np.zeros(q+1)
        for s in range(q+1):
            Omega[s]=(1/(numIterGibbs-burnIn))*np.sum((posterior[s:]-(1/(numIterGibbs-burnIn))*posteriorM)**2)
        varPosterior=Omega[0]
        for s in range(1,q+1):
            varPosterior+=(1-s/(q+1))*2*Omega[s]
        varPosterior=(1/(numIterGibbs-burnIn))*varPosterior
        print(Omega)
        print(varPosterior)
        #Likelihood estimation
        w=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            np.random.shuffle(self.data) #shuffle data
            u=np.zeros(self.n)
            S=np.zeros(self.n,dtype=int)
            N=np.zeros(self.n,dtype=int)
            DPmixTemp=DPmixnorm1d(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
            N[0]+=1
            clusters=np.array([0])
            K=1
            u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
            for i in range(1,self.n):
                logProbs=np.zeros(K+1)
                for k in range(K):
                    logProbs[k]+=np.log(N[clusters[k]]/(i+MStar))
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=clusters[k]#pretend y is in cluster k
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=self.n+1
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int)
                    Ntemp=np.array([DPmixtemp.n])
                    logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                logProbs[K]=logProbs[K]+np.log(MStar/(i+MStar))
                ytemp=np.array([self.data[i]])
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)

                sampledCluster=catDistLogProb(logProbs)
                
                if sampledCluster==K:
                    indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                    N[indexClusterAvailable]+=1
                    clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                    S[i]=indexClusterAvailable
                    K+=1
                    #print(K[t],indexClusterAvailable,N[t],S[t])
                else:
                    #print('assigned to existing cluster',clusters[sampledCluster])
                    S[i]=clusters[sampledCluster]
                    N[clusters[sampledCluster]]+=1
                #print(i,clusters,K)
                
                u[i]=logsumexp(logProbs)
            w[t]=np.sum(u)
        llk=-np.log(numIterGibbs-burnIn)+logsumexp(w)
        logprior=ss.gamma.logpdf(MStar,a=self.priorParamM[0],scale=1/self.priorParamM[1])
        print('llk uncertainty sd', 1/np.sqrt(numIterGibbs-burnIn)*np.sqrt(np.var(np.exp(w)))/np.mean(np.exp(w)))
        print('log posterior', logposteriorM)
        print('llk', llk)
        print('logprior',logprior)
        return llk+logprior-logposteriorM,varPosterior, 1/np.sqrt(numIterGibbs-burnIn)*np.sqrt(np.var(np.exp(w)))/np.mean(np.exp(w))
    
    def ChibEstimator4(self,numIterGibbs,burnIn): #introducing data randomly in the likelihood ordinate estimator
        
        
        #posterior estimation
        S,N,M,K,eta=self.GibbsSamplingClusters(numIterGibbs)
        MStar=np.median(M[burnIn:])
        kde = ss.gaussian_kde(M[burnIn:])
        logposteriorM=np.log(kde.evaluate(MStar)[0])    
        
        #Likelihood estimation
        w=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            np.random.shuffle(self.data) #shuffle data
            u=np.zeros(self.n)
            S=np.zeros(self.n,dtype=int)
            N=np.zeros(self.n,dtype=int)
            DPmixTemp=DPmixnorm1d(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
            N[0]+=1
            clusters=np.array([0])
            K=1
            u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
            for i in range(1,self.n):
                logProbs=np.zeros(K+1)
                for k in range(K):
                    logProbs[k]+=np.log(N[clusters[k]]/(i+MStar))
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=clusters[k]#pretend y is in cluster k
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=self.n+1
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int)
                    Ntemp=np.array([DPmixtemp.n])
                    logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                logProbs[K]=logProbs[K]+np.log(MStar/(i+MStar))
                ytemp=np.array([self.data[i]])
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)

                sampledCluster=catDistLogProb(logProbs)
                
                if sampledCluster==K:
                    indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                    N[indexClusterAvailable]+=1
                    clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                    S[i]=indexClusterAvailable
                    K+=1
                    #print(K[t],indexClusterAvailable,N[t],S[t])
                else:
                    #print('assigned to existing cluster',clusters[sampledCluster])
                    S[i]=clusters[sampledCluster]
                    N[clusters[sampledCluster]]+=1
                #print(i,clusters,K)
                
                u[i]=logsumexp(logProbs)
            w[t]=np.sum(u)
        llk=-np.log(numIterGibbs-burnIn)+logsumexp(w)
        logprior=ss.gamma.logpdf(MStar,a=self.priorParamM[0],scale=1/self.priorParamM[1])
        print('llk uncertainty sd', 1/np.sqrt(numIterGibbs-burnIn)*np.sqrt(np.var(np.exp(w)))/np.mean(np.exp(w)))
        print('log posterior', logposteriorM)
        print('llk', llk)
        print('logprior',logprior)
        return llk+logprior-logposteriorM
    def ChibEstimatorSplitMerge(self,numIterGibbs,numIterLaunch,burnIn):
        
        
        #posterior estimation
        S,N,M,K,eta=self.GibbsSamplingClustersSplitAndMerge(numIterGibbs, numIterLaunch)
        MStar=np.median(M[burnIn:])
        posteriorM=0
        for t in range(burnIn,numIterGibbs):
            #eta=ss.beta.rvs(a=MStar+1,b=self.n)
            epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta[t-1]))+self.priorParamM[0]+K[t]-1)
            #epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta)))
            posteriorM+=(epsilon*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t], scale=1/(self.priorParamM[1]-np.log(eta[t-1])))+(1-epsilon)*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t]-1, scale=1/(self.priorParamM[1]-np.log(eta[t-1]))))
            #print(epsilon)
            
        logposteriorM=-np.log(numIterGibbs-burnIn)+np.log(posteriorM)
            
        #Likelihood estimation
        w=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            u=np.zeros(self.n)
            S=np.zeros(self.n,dtype=int)
            N=np.zeros(self.n,dtype=int)
            DPmixTemp=DPmixnorm1d(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
            N[0]+=1
            clusters=np.array([0])
            K=1
            u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
            for i in range(1,self.n):
                logProbs=np.zeros(K+1)
                for k in range(K):
                    logProbs[k]+=np.log(N[clusters[k]]/(i+MStar))
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=clusters[k]#pretend y is in cluster k
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=self.n+1
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int)
                    Ntemp=np.array([DPmixtemp.n])
                    logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                logProbs[K]=logProbs[K]+np.log(MStar/(i+MStar))
                ytemp=np.array([self.data[i]])
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)

                sampledCluster=catDistLogProb(logProbs)
                
                if sampledCluster==K:
                    indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                    N[indexClusterAvailable]+=1
                    clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                    S[i]=indexClusterAvailable
                    K+=1
                    #print(K[t],indexClusterAvailable,N[t],S[t])
                else:
                    #print('assigned to existing cluster',clusters[sampledCluster])
                    S[i]=clusters[sampledCluster]
                    N[clusters[sampledCluster]]+=1
                #print(i,clusters,K)
                
                u[i]=logsumexp(logProbs)
            w[t]=np.sum(u)
        llk=-np.log(numIterGibbs-burnIn)+logsumexp(w)
        logprior=ss.gamma.logpdf(MStar,a=self.priorParamM[0],scale=1/self.priorParamM[1])
        print('llk uncertainty sd', 1/np.sqrt(numIterGibbs-burnIn)*np.sqrt(np.var(np.exp(w)))/np.mean(np.exp(w)))
        print('log posterior', logposteriorM)
        print('llk', llk)
        print('logprior',logprior)
        return llk+logprior-logposteriorM
    def CollapsedGibbsMoveTempered(self,S0,N0,M,temperature):
        
        clusters=np.unique(S0)
        K=len(clusters)
        #print(clusters)
        #indexClusterAvailable=-1
        for i in range(self.n):
            
            #Remove y_i from the data
            #print('S0',S0)
            #print('N0',N0)
            NminusI=N0.copy()
            NminusI[S0[i]]=NminusI[S0[i]]-1
            if NminusI[S0[i]]==0:
                #print('cluster being emptied')
                #print('NminusI')
                #print(NminusI)
                #print(S0[i])
                #indexClusterAvailable=S0[i]
                K=K-1
                clusters=np.setdiff1d(clusters,np.array([S0[i]]))
                #print("new cluster config", clusters)
            #SminusI=np.delete(S0,i)
            logProbs=np.zeros(K+1)
            for k in range(K):
                logProbs[k]+=np.log(NminusI[clusters[k]])-np.log(self.n-1+M)
                
                Stemp=S0.copy()
                Stemp[i]=clusters[k]#pretend y is in cluster k
                ytemp=self.data[Stemp==clusters[k]]
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.zeros(DPmixtemp.n,dtype=int)
                Ntemp=np.array([DPmixtemp.n])
                logProbs[k]+=DPmixtemp.logLikelihoodAllocationsTempered(Stemp,Ntemp,temperature)
                
                Stemp=S0.copy()
                ytemp=self.data[Stemp==clusters[k]]
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.zeros(DPmixtemp.n,dtype=int)
                Ntemp=np.array([DPmixtemp.n])
                logProbs[k]-=DPmixtemp.logLikelihoodAllocationsTempered(Stemp,Ntemp,temperature)
                
            logProbs[K]+=np.log(M)-np.log(self.n-1+M)
            ytemp=np.array([self.data[i]])
            DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
            Stemp=np.array([0])
            Ntemp=np.array([1])
            logProbs[k]+=DPmixtemp.logLikelihoodAllocationsTempered(Stemp,Ntemp,temperature)
            
            sampledCluster=catDistLogProb(logProbs)
            #print('sampled cluster', sampledCluster)
            # if sampledCluster==K and indexClusterAvailable!=-1:
            #     print('new cluster in the previously emptied cluster')
            #     NminusI[indexClusterAvailable]+=1
            #     clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
            #     S0[i]=indexClusterAvailable
            #     N0=NminusI.copy()
            #     K+=1
            #     indexClusterAvailable=-1
            if sampledCluster==K:
                #print('new cluster created')
                indexClusterAvailable=next((idx for idx, val in np.ndenumerate(NminusI) if val==0))[0] #Find the first zero cluster
                #print(indexClusterAvailable)
                # S0[i]=np.max(S0)+1
                # NminusI[S0[i]]+=1
                # clusters=np.insert(clusters,len(clusters),S0[i])
                # N0=NminusI.copy()
                NminusI[indexClusterAvailable]+=1
                clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                S0[i]=indexClusterAvailable
                N0=NminusI.copy()
                K+=1
                
            else:
                #print('assigned to existing cluster',clusters[sampledCluster])
                S0[i]=clusters[sampledCluster]
                N0=NminusI.copy()
                N0[clusters[sampledCluster]]+=1
            #print('K',K)
            
                
        #Sample alpha
        ##Sample eta, latent var
        #print('K',K)
        eta=ss.beta.rvs(a=M+1,b=self.n)
        epsilon=(self.priorParamM[0]+K-1)/(self.n*(self.priorParamM[1]-np.log(eta))+self.priorParamM[0]+K-1)
        if(np.random.random()<epsilon):
            M=ss.gamma.rvs(a=self.priorParamM[0]+K, scale=1/(self.priorParamM[1]-np.log(eta)))
        else:
            M=ss.gamma.rvs(a=self.priorParamM[0]+K-1, scale=1/(self.priorParamM[1]-np.log(eta)))
                
        #print(clusters)       
        return S0,N0,M
    
    def MHwithinGibbsAllocMove(self,S0,N0,M0,temperature):
        
        clusters=np.unique(S0)
        K=len(clusters)
        
        #Draw the auxiliary variable eta cf https://users.soe.ucsc.edu/~thanos/notes-2.pdf given M0
        eta=ss.beta.rvs(a=M0+1,b=self.n)
        
        #Draw M from its augmented posterior full conditional
        epsilon=(self.priorParamM[0]+K-1)/(self.n*(self.priorParamM[1]-np.log(eta))+self.priorParamM[0]+K-1)
        if(np.random.random()<epsilon):
            M=ss.gamma.rvs(a=self.priorParamM[0]+K, scale=1/(self.priorParamM[1]-np.log(eta)))
        else:
            M=ss.gamma.rvs(a=self.priorParamM[0]+K-1, scale=1/(self.priorParamM[1]-np.log(eta)))
    
        #MH within gibbs step for S
        ##Draw S from its prior distribution given M
        SCand,NCand=self.priorAllocationSim(M)
        if temperature*(self.logLikelihoodAllocations(SCand,NCand)-self.logLikelihoodAllocations(S0,N0))>np.log(np.random.random()):
            S=SCand.copy()
            N=NCand.copy()
        else:
            S=S0.copy()
            N=N0.copy()
        
        return S,N,M
    
    def arithmeticMean(self,numSim):
        llk=np.zeros(numSim)
        for i in range(numSim):
            M=self.priorSimM()
            S,N=self.priorAllocationSim(M)
            llk[i]=self.logLikelihoodAllocations(S, N)
        return -np.log(numSim)+logsumexp(llk)
    
    def harmonicMean(self,numSim,burnIn):
        if numSim<burnIn:
            sys.exit('number of simulations must be bigger than burn in')
        S,N,_,_,_=self.GibbsSamplingClusters(numSim)
        invLogLik=np.zeros(numSim-burnIn)
        for t in range(burnIn,numSim):
            invLogLik[t-burnIn]=-self.logLikelihoodAllocations(S[t], N[t])
        return -(np.log(1/(numSim-burnIn))+logsumexp(invLogLik))
        
    def nestedSamplingAlloc(self,numIterMax,numParticles):
        
        S=[]
        N=[]
        M=[]
        L=np.zeros(numParticles)
        Z=0 #estimate of the normalising constants
        u=[] #successive contributions to the marginal llk
        #Sample from the prior
        if self.priorDistM!='gamma':
            sys.exit('The choice of prior for the concentration parameter M is not supported yet')
        for j in range(numParticles):
            M.append(ss.gamma.rvs(a=self.priorParamM[0],scale=self.priorParamM[1]))
            simS,simN=self.priorAllocationSim(M[-1])
            S.append(simS)
            N.append(simN)
            L[j]=self.logLikelihoodAllocations(S[-1],N[-1])
    

        X=np.zeros(numIterMax) #prior volume
        X[0]=1
        for t in range(1,numIterMax):
            #Computing the prior volume
            X[t]=np.exp(-(t)/numParticles)
            
            
            minIndex=np.argmin(L)
            minL=L[minIndex]
            
            while True:
                Mcand=ss.gamma.rvs(a=self.priorParamM[0],scale=self.priorParamM[1])
                simS,simN=self.priorAllocationSim(Mcand)
                Scand=simS.copy()
                Ncand=simN.copy()
                Lcand=self.logLikelihoodAllocations(Scand,Ncand)
                if Lcand>minL:
                    
                    M[minIndex]=Mcand.copy()
                    S[minIndex]=Scand.copy()
                    N[minIndex]=Ncand.copy()
                    L[minIndex]=Lcand
                    u.append(np.log(X[t-1]-X[t])+minL)
                    Z=logsumexp(u)
                    print(Z)
                    break
            if np.max(L)+np.log(X[t])-Z<0.00001:
                break
        return Z
    
    def SMCFernhead(self,numParticles):
        data=self.data
        w=np.zeros(numParticles)
        for t in range(numParticles):
            np.random.shuffle(data)
            u=np.zeros(self.n)
            S=np.zeros(self.n,dtype=int)
            N=np.zeros(self.n,dtype=int)
            DPmixTemp=DPmixnorm1d(data=data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
            N[0]+=1
            clusters=np.array([0])
            K=1
            M=self.priorSimM()
            u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))#+ss.gamma.logpdf(M,a=self.priorParamM[0],scale=1/self.priorParamM[1])
            for i in range(1,self.n):
                
                logProbs=np.zeros(K+1)
                for k in range(K):
                    logProbs[k]+=np.log(N[clusters[k]]/(i+M))
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=clusters[k]#pretend y is in cluster k
                    ytemp=data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=self.n+1
                    ytemp=data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int)
                    Ntemp=np.array([DPmixtemp.n])
                    logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                logProbs[K]=logProbs[K]+np.log(M/(i+M))
                ytemp=np.array([data[i]])
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)

                sampledCluster=catDistLogProb(logProbs)
                
                if sampledCluster==K:
                    indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                    N[indexClusterAvailable]+=1
                    clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                    S[i]=indexClusterAvailable
                    K+=1
                    #print(K[t],indexClusterAvailable,N[t],S[t])
                else:
                    #print('assigned to existing cluster',clusters[sampledCluster])
                    S[i]=clusters[sampledCluster]
                    N[clusters[sampledCluster]]+=1
                #print(i,clusters,K)
                eta=ss.beta.rvs(a=M+1,b=i+1)
                epsilon=(self.priorParamM[0]+K-1)/((i+1)*(self.priorParamM[1]-np.log(eta))+self.priorParamM[0]+K-1)
                #epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta)))
                if(np.random.random()<epsilon):
                    M=ss.gamma.rvs(a=self.priorParamM[0]+K, scale=1/(self.priorParamM[1]-np.log(eta)))
                else:
                    M=ss.gamma.rvs(a=self.priorParamM[0]+K-1, scale=1/(self.priorParamM[1]-np.log(eta)))
               
                u[i]=logsumexp(logProbs)
            
            w[t]=np.sum(u)
        llk=-np.log(numParticles)+logsumexp(w)
        logvar=-np.log(numParticles)+logsumexp(2*w) + np.log(1-np.exp((-np.log(numParticles**2)+2*logsumexp(w))-(-np.log(numParticles)+logsumexp(2*w))))
        print('var', logvar)
        print('llk', llk)
        return llk,1/np.sqrt(numParticles)*np.exp(0.5*logvar-llk)
                
        

    def NewtonAllocationPosterior(self,S,N,M):
        
        logDensity=ss.gamma.logpdf(M,a=self.priorParamM[0],scale=1/self.priorParamM[1]) #prior log-density
        K=1
        #np.unique without sorting
        indexes = np.unique(S, return_index=True)[1]
        clusters=[S[index] for index in sorted(indexes)]
        Nprogressive=np.zeros(self.n,dtype=int)-1
        Nprogressive[S[0]]=1
       
        for i in range(1,self.n):
            logProbs=np.zeros(K+1)
            newClusterInd=self.n+1
            
            for k in range(K):
                if S[i]==clusters[k]:
                    newClusterInd=k
                logProbs[k]+=np.log(Nprogressive[clusters[k]]/(i+M))
                Stemp=S[:(i+1)].copy()
                Stemp[i]=clusters[k]#pretend y is in cluster k
                ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                Ntemp=np.array([DPmixtemp.n]).copy()
                logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                
                Stemp=S[:(i+1)].copy()
                Stemp[i]=self.n+1
                ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.zeros(DPmixtemp.n,dtype=int)
                Ntemp=np.array([DPmixtemp.n])
                logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                
            logProbs[K]=logProbs[K]+np.log(M/(i+M))
            ytemp=np.array([self.data[i]])
            DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
            Stemp=np.array([0]).copy()
            Ntemp=np.array([1]).copy()
            logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
            
            normalisingConstant=logsumexp(logProbs)
            if newClusterInd==self.n+1: #observation is in a new cluster
                logDensity+=logProbs[K]-normalisingConstant
                K+=1
                Nprogressive[S[i]]=1
                
                
            else: #observation is in cluster newClusterInd=k
                logDensity+=logProbs[newClusterInd]-normalisingConstant
                Nprogressive[clusters[newClusterInd]]+=1
            
                
        return logDensity
    def GeyerEstimator(self,numIter,burnIn,distribution,numSimPrior=-1,NewtonRaphson=False,logisticreg=False):
        
        if distribution=='prior' and numSimPrior!=-1 and numSimPrior>numIter-burnIn and NewtonRaphson==True:
            S1=[]
            N1=[]
            M1=[]
            S2=[]
            N2=[]
            M2=[]
            for i in range(numSimPrior):
                M1.append(self.priorSimM())
                out=self.priorAllocationSim(M1[-1])
                S1.append(out[0])
                N1.append(out[1])
            S2,N2,M2,_,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            M2=M2[burnIn:]
            numSimPosterior=numIter-burnIn
            p11=np.zeros(numSimPrior)
            p21=np.zeros(numSimPrior)
            p12=np.zeros(numSimPosterior)
            p22=np.zeros(numSimPosterior)
            for i in range(numSimPosterior):
                p11[i]=self.logPrior(S1[i],N1[i],M1[i])+np.log(numSimPrior/(numSimPosterior+numSimPrior))
                #p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])
                p12[i]=self.logPrior(S2[i],N2[i],M2[i])+np.log(numSimPrior/(numSimPosterior+numSimPrior))
                #p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])
    
            for i in range(numSimPosterior,numSimPrior):
                p11[i]=self.logPrior(S1[i],N1[i],M1[i])+np.log(numSimPrior/(numSimPosterior+numSimPrior))
                
                    
            def objectiveLogLlk(eta2):
                p1=0
                p2=0
                for i in range(numSimPosterior):
                    p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i],M1[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p1+=p11[i]-logsumexp(np.array([p11[i],p21[i]]))
                    
                    p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p2+=p22[i]-logsumexp(np.array([p12[i],p22[i]]))
            
                for i in range(numSimPosterior,numSimPrior):
                    p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i],M1[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p1+=p11[i]-logsumexp(np.array([p11[i],p21[i]]))
                return(-(p1+p2))
            
            def firstderiv(eta2):
                p1=np.zeros(numSimPrior)
                p2=np.zeros(numSimPosterior)
                for i in range(numSimPosterior):
                    p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i],M1[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p1[i]=p21[i]-logsumexp(np.array([p11[i],p21[i]]))
                    
                    p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p2+=p22[i]-logsumexp(np.array([p12[i],p22[i]]))
            
                for i in range(numSimPosterior,numSimPrior):
                    p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i],M1[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p1[i]=p21[i]-logsumexp(np.array([p11[i],p21[i]]))
                return numSimPosterior-np.sum(np.exp(p1))-np.sum(np.exp(p2))
            
            def secondderiv(eta2):
                p1=np.zeros(numSimPrior)
                p2=np.zeros(numSimPosterior)
                for i in range(numSimPosterior):
                    p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i],M1[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p1[i]=p21[i]-logsumexp(np.array([p11[i],p21[i]]))
                    
                    p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p2+=p22[i]-logsumexp(np.array([p12[i],p22[i]]))
            
                for i in range(numSimPosterior,numSimPrior):
                    p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i],M1[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p1[i]=p21[i]-logsumexp(np.array([p11[i],p21[i]]))
                
                return -np.sum(np.exp(p1)*(1-np.exp(p1)))-np.sum(np.exp(p2)*(1-np.exp(p2)))
            
            # x=[]
            # x.append(24)
            # while True :
            #     x.append(x[-1]-firstderiv(x[-1])/secondderiv(x[-1]))
            #     print(x[-1])
            # def objectiveLogLlk(eta2):
            #     p1=0
            #     p2=0
            #     for i in range(numIter-burnIn):
            #         p11=self.logPrior(S1[i],N1[i])
            #         p21=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])+eta2
            #         p1+=p11-logsumexp(np.array([p11,p21]))
                    
            #         p12=self.logPrior(S2[i],N2[i])
            #         p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])+eta2
            #         p2+=p22-logsumexp(np.array([p12,p22]))
            #     return(-(p1+p2))
            res=minimize(objectiveLogLlk, 22, method='Newton-CG',jac=firstderiv, hess=secondderiv,options={'xtol': 1e-8, 'disp': True})
            return print(res)
        if distribution=='prior' and numSimPrior!=-1 and numSimPrior>numIter-burnIn: ###attention j'ai oublié de mettre les poids T1/(T1+T2) etc
            S1=[]
            N1=[]
            M1=[]
            S2=[]
            N2=[]
            M2=[]
            w1=numSimPrior/(numSimPrior+numIter-burnIn)
            w2=1-w1
            for i in range(numSimPrior):
                M1.append(self.priorSimM())
                out=self.priorAllocationSim(M1[-1])
                S1.append(out[0])
                N1.append(out[1])
            S2,N2,M2,_,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            M2=M2[burnIn:]
            numSimPosterior=numIter-burnIn
            p11=np.zeros(numSimPrior)
            p21=np.zeros(numSimPrior)
            p12=np.zeros(numSimPosterior)
            p22=np.zeros(numSimPosterior)
            for i in range(numSimPosterior):
                p11[i]=self.logPrior(S1[i],N1[i],M1[i])+np.log(numSimPrior/(numSimPosterior+numSimPrior))
                #p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])
                p12[i]=self.logPrior(S2[i],N2[i],M2[i])+np.log(numSimPrior/(numSimPosterior+numSimPrior))
                #p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])
    
            for i in range(numSimPosterior,numSimPrior):
                p11[i]=self.logPrior(S1[i],N1[i],M1[i])+np.log(numSimPrior/(numSimPosterior+numSimPrior))
                
                    
            def objectiveLogLlk(eta2):
                p1=0
                p2=0
                for i in range(numSimPosterior):
                    p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i],M1[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p1+=np.log(w1)+p11[i]-logsumexp(np.array([np.log(w1)+p11[i],np.log(w2)+p21[i]]))
                    
                    p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p2+=np.log(w2)+p22[i]-logsumexp(np.array([np.log(w1)+p12[i],np.log(w2)+p22[i]]))
            
                for i in range(numSimPosterior,numSimPrior):
                    p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i],M1[i])+np.log(numSimPosterior/(numSimPosterior+numSimPrior))+eta2
                    p1+=p11[i]-logsumexp(np.array([p11[i],p21[i]]))
                return(-(p1+p2))
            # def objectiveLogLlk(eta2):
            #     p1=0
            #     p2=0
            #     for i in range(numIter-burnIn):
            #         p11=self.logPrior(S1[i],N1[i])
            #         p21=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])+eta2
            #         p1+=p11-logsumexp(np.array([p11,p21]))
                    
            #         p12=self.logPrior(S2[i],N2[i])
            #         p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])+eta2
            #         p2+=p22-logsumexp(np.array([p12,p22]))
            #     return(-(p1+p2))
            res=minimize_scalar(objectiveLogLlk)
            return -res.x
        if distribution=='prior' and numSimPrior!=-1 and numSimPrior<numIter-burnIn:
            sys.exit('Not supported yet : number of simulations from prior must be greater than number of simulations from posterior')
        if distribution=='prior':
            S1=[]
            N1=[]
            M1=[]
            S2=[]
            N2=[]
            M2=[]
            for i in range(numIter-burnIn):
                M1.append(self.priorSimM())
                out=self.priorAllocationSim(M1[-1])
                S1.append(out[0])
                N1.append(out[1])
            S2,N2,M2,_,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            M2=M2[burnIn:]
            
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            p11=np.zeros(numIter-burnIn)
            p21=np.zeros(numIter-burnIn)
            p12=np.zeros(numIter-burnIn)
            p22=np.zeros(numIter-burnIn)
            for i in range(numIter-burnIn):
                p11[i]=self.logPrior(S1[i],N1[i],M1[i])
                #p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])
                p12[i]=self.logPrior(S2[i],N2[i],M2[i])
                #p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])
            def objectiveLogLlk(eta2):
                p1=0
                p2=0
                for i in range(numIter-burnIn):
                    p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i],M1[i])+eta2
                    p1+=p11[i]-logsumexp(np.array([p11[i],p21[i]]))
                    
                    p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])+eta2
                    p2+=p22[i]-logsumexp(np.array([p12[i],p22[i]]))
                return(-(p1+p2))
            # def objectiveLogLlk(eta2):
            #     p1=0
            #     p2=0
            #     for i in range(numIter-burnIn):
            #         p11=self.logPrior(S1[i],N1[i])
            #         p21=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])+eta2
            #         p1+=p11-logsumexp(np.array([p11,p21]))
                    
            #         p12=self.logPrior(S2[i],N2[i])
            #         p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])+eta2
            #         p2+=p22-logsumexp(np.array([p12,p22]))
            #     return(-(p1+p2))
            res=minimize_scalar(objectiveLogLlk)
            return -res.x
        if distribution=='prior+Newton':
            S0=[]
            N0=[]
            S1=[]
            N1=[]
            S2=[]
            N2=[]
            
            ##Simulating the kind of approximate posterior distribution (like Newton algo) from Chib
            
            for t in range(numIter-burnIn):
                #np.random.shuffle(self.data) #shuffle data
                u=np.zeros(self.n)
                S=np.zeros(self.n,dtype=int)
                N=np.zeros(self.n,dtype=int)
                DPmixTemp=DPmixnorm1dKnownM(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                N[0]+=1
                clusters=np.array([0])
                K=1
                u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
                for i in range(1,self.n):
                    logProbs=np.zeros(K+1)
                    for k in range(K):
                        logProbs[k]+=np.log(N[clusters[k]]/(i+self.M))
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=clusters[k]#pretend y is in cluster k
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=self.n+1
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int)
                        Ntemp=np.array([DPmixtemp.n])
                        logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    logProbs[K]=logProbs[K]+np.log(self.M/(i+self.M))
                    ytemp=np.array([self.data[i]])
                    DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                    Stemp=np.array([0]).copy()
                    Ntemp=np.array([1]).copy()
                    logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    sampledCluster=catDistLogProb(logProbs)
                    if sampledCluster==K:
                        indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                        N[indexClusterAvailable]+=1
                        clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                        S[i]=indexClusterAvailable
                        K+=1
                        #print(K[t],indexClusterAvailable,N[t],S[t])
                    else:
                        #print('assigned to existing cluster',clusters[sampledCluster])
                        S[i]=clusters[sampledCluster]
                        N[clusters[sampledCluster]]+=1
                S0.append(S.copy())
                N0.append(N.copy())
                    
            #Simulating from prior
            for i in range(numIter-burnIn):
                out=self.priorAllocationSim()
                S1.append(out[0])
                N1.append(out[1])
            #Simulating from posterior
            S2,N2,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            p00=np.zeros(numIter-burnIn)
            p10=np.zeros(numIter-burnIn)
            #p20=np.zeros(numIter-burnIn)
            p01=np.zeros(numIter-burnIn)
            p11=np.zeros(numIter-burnIn)
            #p21=np.zeros(numIter-burnIn)
            p02=np.zeros(numIter-burnIn)
            p12=np.zeros(numIter-burnIn)
            #p22=np.zeros(numIter-burnIn)
            for i in range(numIter-burnIn):
                p00[i]=self.NewtonAllocationPosterior(S0[i], N0[i])
                p10[i]=self.logPrior(S0[i],N0[i])
                #p20[i]=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i])
                
                p01[i]=self.NewtonAllocationPosterior(S1[i], N1[i])
                p11[i]=self.logPrior(S1[i],N1[i])
                #p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])
                
                p02[i]=self.NewtonAllocationPosterior(S2[i], N2[i])
                p12[i]=self.logPrior(S2[i],N2[i])
                #p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])
                
            # def objectiveLogLlk(eta2):
            #     p0=0
            #     p1=0
            #     p2=0
            #     for i in range(numIter-burnIn):
                    
            #         p20=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i])+eta2
            #         p0+=p00[i]-logsumexp(np.array([p00[i],p10[i],p20]))
                    
            #         p21=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])+eta2
            #         p1+=p11[i]-logsumexp(np.array([p01[i],p11[i],p21]))
                    
                    
            #         p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])+eta2
            #         p2+=p22-logsumexp(np.array([p02[i],p12[i],p22]))
            #     return(-(p0+p1+p2))
            # res=minimize_scalar(objectiveLogLlk)
            return -res.x
        if distribution=='Newton' and logisticreg==False and numSimPrior==-1:
            S0=[]
            N0=[]
            M0=[]

            
            S2=[]
            N2=[]
            M2=[]
            ##Simulating the kind of approximate posterior distribution (like Newton algo) from Chib
            p00=np.zeros(numIter-burnIn)
            for t in range(numIter-burnIn):
                #np.random.shuffle(self.data) #shuffle data
                M0.append(self.priorSimM())
                u=np.zeros(self.n)
                S=np.zeros(self.n,dtype=int)
                N=np.zeros(self.n,dtype=int)
                DPmixTemp=DPmixnorm1d(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                N[0]+=1
                clusters=np.array([0])
                K=1
                u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
                for i in range(1,self.n):
                    logProbs=np.zeros(K+1)
                    for k in range(K):
                        logProbs[k]+=np.log(N[clusters[k]]/(i+M0[-1]))
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=clusters[k]#pretend y is in cluster k
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=self.n+1
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int)
                        Ntemp=np.array([DPmixtemp.n])
                        logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    logProbs[K]=logProbs[K]+np.log(M0[-1]/(i+M0[-1]))
                    ytemp=np.array([self.data[i]])
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.array([0]).copy()
                    Ntemp=np.array([1]).copy()
                    logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    sampledCluster=catDistLogProb(logProbs)
                    if sampledCluster==K:
                        indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                        N[indexClusterAvailable]+=1
                        clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                        S[i]=indexClusterAvailable
                        K+=1
                        #print(K[t],indexClusterAvailable,N[t],S[t])
                    else:
                        #print('assigned to existing cluster',clusters[sampledCluster])
                        S[i]=clusters[sampledCluster]
                        N[clusters[sampledCluster]]+=1
                        
                    p00[t]+=logProbs[sampledCluster]-logsumexp(logProbs)
                p00[t]+=ss.gamma.logpdf(M0[-1],a=self.priorParamM[0],scale=1/self.priorParamM[1]) #prior log-density
                S0.append(S.copy())
                N0.append(N.copy())
                    
            
            #Simulating from posterior
            S2,N2,M2,_,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            M2=M2[burnIn:]
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            

            p20=np.zeros(numIter-burnIn)
          
            
            p02=np.zeros(numIter-burnIn)
            p22=np.zeros(numIter-burnIn)
            
            for i in range(numIter-burnIn):
                #p00[i]=self.NewtonAllocationPosterior(S0[i], N0[i],M0[i])
                p20[i]=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i],M0[i])
                
          
                p02[i]=self.NewtonAllocationPosterior(S2[i], N2[i],M2[i])
                p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])
                
                
            def objectiveLogLlk(eta2):
                p0=0
                p2=0
                for i in range(numIter-burnIn):
                    
                    #p20=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i],M0[i])+eta2
                    #p0+=p00[i]-logsumexp(np.array([p00[i],p20]))
                    
                  
                    
                    p0+=p00[i]-logsumexp(np.array([p00[i],p20[i]+eta2]))
                    
                    #p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])+eta2
                    #p2+=p22-logsumexp(np.array([p02[i],p22]))
                    p2+=p22[i]+eta2-logsumexp(np.array([p02[i],p22[i]+eta2]))
                return(-(p0+p2))
            res=minimize_scalar(objectiveLogLlk)
            return -res.x
        
        if distribution=='Newton' and logisticreg==False and numSimPrior!=-1:
            S0=[]
            N0=[]
            M0=[]

            
            S2=[]
            N2=[]
            M2=[]
            
            w0=numSimPrior/(numSimPrior+numIter-burnIn)
            w2=1-w0
            ##Simulating the kind of approximate posterior distribution (like Newton algo) from Chib
            p00=np.zeros(numSimPrior)
            for t in range(numSimPrior):
                #np.random.shuffle(self.data) #shuffle data
                M0.append(self.priorSimM())
                u=np.zeros(self.n)
                S=np.zeros(self.n,dtype=int)
                N=np.zeros(self.n,dtype=int)
                DPmixTemp=DPmixnorm1d(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                N[0]+=1
                clusters=np.array([0])
                K=1
                u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
                for i in range(1,self.n):
                    logProbs=np.zeros(K+1)
                    for k in range(K):
                        logProbs[k]+=np.log(N[clusters[k]]/(i+M0[-1]))
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=clusters[k]#pretend y is in cluster k
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=self.n+1
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int)
                        Ntemp=np.array([DPmixtemp.n])
                        logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    logProbs[K]=logProbs[K]+np.log(M0[-1]/(i+M0[-1]))
                    ytemp=np.array([self.data[i]])
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.array([0]).copy()
                    Ntemp=np.array([1]).copy()
                    logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    sampledCluster=catDistLogProb(logProbs)
                    if sampledCluster==K:
                        indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                        N[indexClusterAvailable]+=1
                        clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                        S[i]=indexClusterAvailable
                        K+=1
                        #print(K[t],indexClusterAvailable,N[t],S[t])
                    else:
                        #print('assigned to existing cluster',clusters[sampledCluster])
                        S[i]=clusters[sampledCluster]
                        N[clusters[sampledCluster]]+=1
                        
                    p00[t]+=logProbs[sampledCluster]-logsumexp(logProbs)
                p00[t]+=ss.gamma.logpdf(M0[-1],a=self.priorParamM[0],scale=1/self.priorParamM[1]) #prior log-density
                S0.append(S.copy())
                N0.append(N.copy())
                    
            
            #Simulating from posterior
            S2,N2,M2,_,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            M2=M2[burnIn:]
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            

            p20=np.zeros(numSimPrior)
          
            
            p02=np.zeros(numIter-burnIn)
            p22=np.zeros(numIter-burnIn)
            
            if (numIter-burnIn)>numSimPrior :
                for i in range(numSimPrior):
                    #p00[i]=self.NewtonAllocationPosterior(S0[i], N0[i],M0[i])
                    p20[i]=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i],M0[i])
                
          
                    p02[i]=self.NewtonAllocationPosterior(S2[i], N2[i],M2[i])
                    p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])
                
                for i in range(numSimPrior,numIter-burnIn):
                    p02[i]=self.NewtonAllocationPosterior(S2[i], N2[i],M2[i])
                    p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])
            else :
                sys.exit('The number of simulations of posterior should be greater than that of SIS')
                
            def objectiveLogLlk(eta2):
                p0=0
                p2=0
                for i in range(numSimPrior):
                    
                    #p20=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i],M0[i])+eta2
                    #p0+=p00[i]-logsumexp(np.array([p00[i],p20]))
                    
                  
                    
                    p0+=np.log(w0)+p00[i]-logsumexp(np.array([np.log(w0)+p00[i],np.log(w2)+p20[i]+eta2]))
                    
                    #p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])+eta2
                    #p2+=p22-logsumexp(np.array([p02[i],p22]))
                    p2+=np.log(w2)+p22[i]+eta2-logsumexp(np.array([np.log(w0)+p02[i],np.log(w2)+p22[i]+eta2]))
                for i in range(numSimPrior,numIter-burnIn):
                    p2+=np.log(w2)+p22[i]+eta2-logsumexp(np.array([np.log(w0)+p02[i],np.log(w2)+p22[i]+eta2]))
                return(-(p0+p2))
            res=minimize_scalar(objectiveLogLlk)
            return -res.x
        if distribution=='Newton' and logisticreg==True:
            print('hello')
            S0=[]
            N0=[]
            M0=[]

            
            S2=[]
            N2=[]
            M2=[]
            ##Simulating the kind of approximate posterior distribution (like Newton algo) from Chib
            
            for t in range(numIter-burnIn):
                #np.random.shuffle(self.data) #shuffle data
                M0.append(self.priorSimM())
                u=np.zeros(self.n)
                S=np.zeros(self.n,dtype=int)
                N=np.zeros(self.n,dtype=int)
                DPmixTemp=DPmixnorm1d(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                N[0]+=1
                clusters=np.array([0])
                K=1
                u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
                for i in range(1,self.n):
                    logProbs=np.zeros(K+1)
                    for k in range(K):
                        logProbs[k]+=np.log(N[clusters[k]]/(i+M0[-1]))
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=clusters[k]#pretend y is in cluster k
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=self.n+1
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int)
                        Ntemp=np.array([DPmixtemp.n])
                        logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    logProbs[K]=logProbs[K]+np.log(M0[-1]/(i+M0[-1]))
                    ytemp=np.array([self.data[i]])
                    DPmixtemp=DPmixnorm1d(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.array([0]).copy()
                    Ntemp=np.array([1]).copy()
                    logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    sampledCluster=catDistLogProb(logProbs)
                    if sampledCluster==K:
                        indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                        N[indexClusterAvailable]+=1
                        clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                        S[i]=indexClusterAvailable
                        K+=1
                        #print(K[t],indexClusterAvailable,N[t],S[t])
                    else:
                        #print('assigned to existing cluster',clusters[sampledCluster])
                        S[i]=clusters[sampledCluster]
                        N[clusters[sampledCluster]]+=1
                
                S0.append(S.copy())
                N0.append(N.copy())
                    
            
            #Simulating from posterior
            S2,N2,M2,_,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            M2=M2[burnIn:]
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            p00=np.zeros(numIter-burnIn)

            p20=np.zeros(numIter-burnIn)
          
            
            p02=np.zeros(numIter-burnIn)
            p22=np.zeros(numIter-burnIn)
            for i in range(numIter-burnIn):
                p00[i]=self.NewtonAllocationPosterior(S0[i], N0[i],M0[i])
                p20[i]=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i],M0[i])
                
          
                p02[i]=self.NewtonAllocationPosterior(S2[i], N2[i],M2[i])
                p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])
                
            response=np.concatenate([np.ones(numIter-burnIn),np.zeros(numIter-burnIn)])
            logh1=np.concatenate([p20,p22])
            logh2=np.concatenate([p00,p02])
            predictor=np.column_stack([logh1,logh2])
            clf = LogisticRegression(max_iter=1000,penalty='none',tol=1e-6).fit(predictor, response)

            print(clf.intercept_)
            return clf.intercept_
    
class DPmixnorm1dLocation:
    
    
    def __init__(self, data=None, sigma=None, priorDistLoc=None,
                 priorParamLoc=None, priorDistM=None, priorParamM=None):
        if data is None:
            sys.exit('please specify data')
        else:
            self.data=data
        self.n=data.shape[0]
        
        if sigma is None:
            sys.exit('Please specify some value for the scale parameter sigma')
        else:
            self.sigma=sigma

        if priorDistLoc is None:
            self.priorDistLoc='Conjugate'
        else:
            self.priorDistLoc=priorDistLoc

        if priorParamLoc is None:
            if self.priorDistLoc=='Conjugate':
                self.priorParamLoc=np.array([np.mean(self.data),2.6/(np.max(self.data)-np.min(self.data))]) #b0, N0, c0, C0 according to p.178 Fruhwirth Schnatter, as chsen by raftery 1996
        else:
            self.priorParamLoc=priorParamLoc

        if priorDistM is None:
            self.priorDistM='gamma'
        else:
            self.priorDistM=priorDistM

        if priorParamM is None:
            if self.priorDistM=='gamma':
                self.priorParamM=np.array([1,1]) #Shape and SCALE : watch out with def of wiki and scipy
        else:
            self.priorParamM=priorParamM
    def priorSimM(self):
        if self.priorDistM=='gamma':
            return ss.gamma.rvs(a=self.priorParamM[0],scale=1/self.priorParamM[1]) #Shape and RATE : watch out with def of wiki and scipy
    
    def logLikelihoodAllocations(self,S,N):
        clusters=np.unique(S)
        llk=0
        for k in range(len(clusters)):
            mu0=self.priorParamLoc[0]
            sigma0=self.priorParamLoc[1]
            nk=N[clusters[k]]
            yk=self.data[S==clusters[k]]
            llk+=np.log(self.sigma)-0.5*nk*np.log(2*np.pi*self.sigma**2)-0.5*np.log(nk*sigma0**2+self.sigma**2)
            #print(llk)
            llk+=(-1/(2*self.sigma**2))*np.sum(yk**2)-mu0**2/(2*sigma0**2)
            #print(llk)
            llk+=(1/(2*(nk*sigma0**2+self.sigma**2)))*(((sigma0**2*np.sum(yk)**2)/self.sigma**2)+(self.sigma**2*mu0**2/sigma0**2)+(2*mu0*np.sum(yk)))
        print(llk)
        return llk   
    
    def priorAllocationSim(self,M):
        S=np.zeros(self.n, dtype='int')
        N=np.zeros(self.n, dtype='int')
        N[0]=1
        for i in range(1,self.n):
            if M/(M+i)>np.random.random():
                S[i]=np.max(S)+1
                N[S[i]]+=1
            else:
                newIndex=np.random.choice(self.n,p=N/i)
                S[i]=newIndex
                N[newIndex]+=1
        return(S,N)
    
    def logPrior(self,S,N,M):
        clusters=np.unique(S)
        logPrior=ss.gamma.logpdf(M,a=self.priorParamM[0],scale=1/self.priorParamM[1])
        logPrior+=len(clusters)*np.log(M)+loggamma(M)-loggamma(M+self.n)
        for k in range(len(clusters)):
            logPrior+=loggamma(N[clusters[k]])
        return logPrior
    
    def GibbsSamplingClusters(self,numIter):
        S=[]
        N=[]
        M=[]
        eta=[] #augmentation variable to compute posterior full conditional on M
        K=np.zeros(numIter,dtype=int)
        
        #intialise from the prior
        S.append(np.zeros(self.n,dtype=int))
        N.append(np.zeros(self.n,dtype=int))
        M.append(self.priorSimM())
        S[0],N[0]=self.priorAllocationSim(M[0])
        clusters=np.unique(S[0])
        K[0]=len(clusters)
        
        for t in range(1,numIter):
            S.append(np.zeros(self.n,dtype=int))
            S[t]=S[t-1].copy()
            N.append(np.zeros(self.n,dtype=int))
            N[t]=N[t-1].copy()
            M.append(0)
            M[t]=M[t-1].copy()
            K[t]=K[t-1].copy()
            
            for i in range(self.n):
                
                #Remove y_i from the data
                NminusI=N[t].copy()
                NminusI[S[t][i]]=NminusI[S[t][i]]-1
                if NminusI[S[t][i]]==0:
                    #A cluster is being emptied
                    K[t]=K[t]-1
                    clusters=np.setdiff1d(clusters,np.array([S[t][i]]))
                logProbs=np.zeros(K[t]+1)
                for k in range(K[t]):
                    
                    logProbs[k]=logProbs[k]+np.log(NminusI[clusters[k]]/(self.n-1+M[t]))
                    Stemp=S[t].copy()
                    Stemp[i]=clusters[k]#pretend y is in cluster k
                    ytemp=self.data[Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma, priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[t].copy()
                    Stemp[i]=self.n+1
                    ytemp=self.data[Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int)
                    Ntemp=np.array([DPmixtemp.n])
                    logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                 
                    
                
                logProbs[K[t]]=logProbs[K[t]]+np.log(M[t]/(self.n-1+M[t]))
                ytemp=np.array([self.data[i]])
                DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                
                logProbs[K[t]]=logProbs[K[t]]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)

                sampledCluster=catDistLogProb(logProbs)
                
                if sampledCluster==K[t]:
                    indexClusterAvailable=next((idx for idx, val in np.ndenumerate(NminusI) if val==0))[0] #Find the first zero cluster
                    NminusI[indexClusterAvailable]+=1
                    clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                    S[t][i]=indexClusterAvailable
                    N[t]=NminusI.copy()
                    K[t]+=1
                    #print(K[t],indexClusterAvailable,N[t],S[t])
                else:
                    #print('assigned to existing cluster',clusters[sampledCluster])
                    S[t][i]=clusters[sampledCluster]
                    NminusI[clusters[sampledCluster]]+=1
                    N[t]=NminusI.copy()
                

            #Sample concentration parameter M
            ##Sample eta, latent var
            #print('K',K)
            eta.append(ss.beta.rvs(a=M[t]+1,b=self.n))
            epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta[-1]))+self.priorParamM[0]+K[t]-1)
            #epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta)))
            if(np.random.random()<epsilon):
                M[t]=ss.gamma.rvs(a=self.priorParamM[0]+K[t], scale=1/(self.priorParamM[1]-np.log(eta[-1])))
            else:
                M[t]=ss.gamma.rvs(a=self.priorParamM[0]+K[t]-1, scale=1/(self.priorParamM[1]-np.log(eta[-1])))
        
        return(S,N,M,K,eta)
    
    def adaptiveSMCAlloc2(self,numParticles,numGibbsMove,ESSThreshold,maxIterBissection=10000,TOL=0.000000005):
        temperatures=[]
        temperatures.append(0)

        
        S=[]
        N=[]
        M=[]
        Z=[]
        Z.append(0)#estimate of the successive ratios of normalising constants
        #Sample from the prior
        if self.priorDistM!='gamma':
            sys.exit('The choice of prior for the concentration parameter M is not supported yet')
        #print('prior particules')
        for j in range(numParticles):
            M.append(ss.gamma.rvs(a=self.priorParamM[0],scale=self.priorParamM[1]))
            simS,simN=self.priorAllocationSim(M[-1])
            S.append(simS)
            N.append(simN)
            W=np.log(np.ones(numParticles)/numParticles)
            w=np.log(np.ones(numParticles)/numParticles)
        
        k=0
        while temperatures[k]<1:
            #Find the next temperature adaptively
            llkParticles=np.zeros(numParticles)
            for j in range(numParticles):
                    llkParticles[j]=self.logLikelihoodAllocations(S[j],N[j])
            #Try temperature = 1
            wtemp=np.zeros(numParticles)
            for j in range(numParticles):
                wtemp[j]=(1-temperatures[k])*llkParticles[j]
            Wtemp=wtemp-logsumexp(wtemp)
            logESS=-logsumexp(2*Wtemp)
            if logESS>np.log(ESSThreshold*numParticles):
                temperatures.append(1)
            #else do bissection algorithm
            else:
                print('statr')
                a=temperatures[k]
                b=1
                l=0
                while l<maxIterBissection:
                    tempcand=(a+b)/2
                    print(tempcand)
                    for j in range(numParticles):
                        wtemp[j]=(tempcand-temperatures[k])*llkParticles[j]
                    Wtemp=wtemp-logsumexp(wtemp)
                    logESS=-logsumexp(2*Wtemp)
                    if np.abs(logESS-np.log(ESSThreshold*numParticles))<np.log(TOL) or ((b-a)/2)<TOL:
                        break
                    else:
                        if logESS>np.log(ESSThreshold*numParticles): #need to increase the temp
                            a=tempcand
                        else:
                            b=tempcand
                    l=l+1
                    
                temperatures.append(tempcand)
                print(l)
            print(temperatures[k],np.exp(logESS))
            k=k+1
            for j in range(numParticles):
                #Reweighting
                w[j]=W[j]+(temperatures[k]-temperatures[k-1])*llkParticles[j]
            Z.append(logsumexp(w))
            #Renormalising the log weights
            W=w-Z[k]
            #Resample
            logESS=-logsumexp(2*W)
            print(np.exp(logESS))
            if logESS<np.log(0.8*numParticles):#Resample multinomially
                #print('resampling')
                Scopy=S.copy()
                Ncopy=N.copy()
                Mcopy=M.copy()
                for j in range(numParticles):
                    #print('Particule before :')
                    #print(S[j])
                    sampledIndex=catDistLogProb(w)
                    S[j]=Scopy[sampledIndex].copy()
                    N[j]=Ncopy[sampledIndex].copy()
                    M[j]=Mcopy[sampledIndex].copy()
                    #print('Particule after :')
                    #print(S[j])
                W=np.log(np.ones(numParticles)/numParticles)
            #Mutate
            Scopy=S.copy()
            Ncopy=N.copy()
            Mcopy=M.copy()    
            for j in range(numParticles):
                
                for i in range(numGibbsMove):
                
                    Scopy[j],Ncopy[j],Mcopy[j]=self.MHwithinGibbsAllocMove(Scopy[j],Ncopy[j],Mcopy[j],temperatures[k])
                    
                #print('moved particle')
                #print(S[j],N[j],M[j])
                S[j]=Scopy[j].copy()
                N[j]=Ncopy[j].copy()
                M[j]=Mcopy[j].copy()
            
        #print(S)
        #print(N)
        #print(M)
        print(np.sum(Z))
        return(np.sum(Z))
    
    def MHwithinGibbsAllocMove(self,S0,N0,M0,temperature):
        
        clusters=np.unique(S0)
        K=len(clusters)
        
        #Draw the auxiliary variable eta cf https://users.soe.ucsc.edu/~thanos/notes-2.pdf given M0
        eta=ss.beta.rvs(a=M0+1,b=self.n)
        
        #Draw M from its augmented posterior full conditional
        epsilon=(self.priorParamM[0]+K-1)/(self.n*(self.priorParamM[1]-np.log(eta))+self.priorParamM[0]+K-1)
        if(np.random.random()<epsilon):
            M=ss.gamma.rvs(a=self.priorParamM[0]+K, scale=1/(self.priorParamM[1]-np.log(eta)))
        else:
            M=ss.gamma.rvs(a=self.priorParamM[0]+K-1, scale=1/(self.priorParamM[1]-np.log(eta)))
    
        #MH within gibbs step for S
        ##Draw S from its prior distribution given M
        SCand,NCand=self.priorAllocationSim(M)
        if temperature*(self.logLikelihoodAllocations(SCand,NCand)-self.logLikelihoodAllocations(S0,N0))>np.log(np.random.random()):
            S=SCand.copy()
            N=NCand.copy()
        else:
            S=S0.copy()
            N=N0.copy()
        
        return S,N,M
    
    def arithmeticMean(self,numSim):
        llk=np.zeros(numSim)
        for i in range(numSim):
            M=self.priorSimM()
            S,N=self.priorAllocationSim(M)
            llk[i]=self.logLikelihoodAllocations(S, N)
        return -np.log(numSim)+logsumexp(llk)
    
    def harmonicMean(self,numSim,burnIn):
        if numSim<burnIn:
            sys.exit('number of simulations must be bigger than burn in')
        S,N,_,_,_=self.GibbsSamplingClusters(numSim)
        invLogLik=np.zeros(numSim-burnIn)
        for t in range(burnIn,numSim):
            invLogLik[t-burnIn]=-self.logLikelihoodAllocations(S[t], N[t])
        return -(np.log(1/(numSim-burnIn))+logsumexp(invLogLik))
    
    def ChibEstimator2(self,numIterGibbs,burnIn):
        
        
        #posterior estimation
        S,N,M,K,eta=self.GibbsSamplingClusters(numIterGibbs)
        MStar=np.median(M[burnIn:])
        posteriorM=0
        for t in range(burnIn,numIterGibbs):
            #eta=ss.beta.rvs(a=MStar+1,b=self.n)
            epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta[t-1]))+self.priorParamM[0]+K[t]-1)
            #epsilon=(self.priorParamM[0]+K[t]-1)/(self.n*(self.priorParamM[1]-np.log(eta)))
            posteriorM+=(epsilon*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t], scale=1/(self.priorParamM[1]-np.log(eta[t-1])))+(1-epsilon)*ss.gamma.pdf(MStar,a=self.priorParamM[0]+K[t]-1, scale=1/(self.priorParamM[1]-np.log(eta[t-1]))))
            #print(epsilon)
            
        logposteriorM=-np.log(numIterGibbs-burnIn)+np.log(posteriorM)
            
        #Likelihood estimation
        w=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            u=np.zeros(self.n)
            S=np.zeros(self.n,dtype=int)
            N=np.zeros(self.n,dtype=int)
            DPmixTemp=DPmixnorm1dLocation(data=self.data[0],priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
            N[0]+=1
            clusters=np.array([0])
            K=1
            u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
            for i in range(1,self.n):
                logProbs=np.zeros(K+1)
                for k in range(K):
                    logProbs[k]+=np.log(N[clusters[k]]/(i+MStar))
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=clusters[k]#pretend y is in cluster k
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=self.n+1
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int)
                    Ntemp=np.array([DPmixtemp.n])
                    logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                logProbs[K]=logProbs[K]+np.log(MStar/(i+MStar))
                ytemp=np.array([self.data[i]])
                DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)

                sampledCluster=catDistLogProb(logProbs)
                
                if sampledCluster==K:
                    indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                    N[indexClusterAvailable]+=1
                    clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                    S[i]=indexClusterAvailable
                    K+=1
                    #print(K[t],indexClusterAvailable,N[t],S[t])
                else:
                    #print('assigned to existing cluster',clusters[sampledCluster])
                    S[i]=clusters[sampledCluster]
                    N[clusters[sampledCluster]]+=1
                #print(i,clusters,K)
                
                u[i]=logsumexp(logProbs)
            w[t]=np.sum(u)
        llk=-np.log(numIterGibbs-burnIn)+logsumexp(w)
        logprior=ss.gamma.logpdf(MStar,a=self.priorParamM[0],scale=1/self.priorParamM[1])
        print('llk uncertainty sd', 1/np.sqrt(numIterGibbs-burnIn)*np.sqrt(np.var(np.exp(w)))/np.mean(np.exp(w)))
        print('log posterior', logposteriorM)
        print('llk', llk)
        print('logprior',logprior)
        return llk+logprior-logposteriorM
    
    def NewtonAllocationPosterior(self,S,N,M):
        
        logDensity=ss.gamma.logpdf(M,a=self.priorParamM[0],scale=1/self.priorParamM[1]) #prior log-density
        K=1
        #np.unique without sorting
        indexes = np.unique(S, return_index=True)[1]
        clusters=[S[index] for index in sorted(indexes)]
        Nprogressive=np.zeros(self.n,dtype=int)-1
        Nprogressive[S[0]]=1
       
        for i in range(1,self.n):
            logProbs=np.zeros(K+1)
            newClusterInd=self.n+1
            
            for k in range(K):
                if S[i]==clusters[k]:
                    newClusterInd=k
                logProbs[k]+=np.log(Nprogressive[clusters[k]]/(i+M))
                Stemp=S[:(i+1)].copy()
                Stemp[i]=clusters[k]#pretend y is in cluster k
                ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                Ntemp=np.array([DPmixtemp.n]).copy()
                logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                
                Stemp=S[:(i+1)].copy()
                Stemp[i]=self.n+1
                ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                Stemp=np.zeros(DPmixtemp.n,dtype=int)
                Ntemp=np.array([DPmixtemp.n])
                logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                
            logProbs[K]=logProbs[K]+np.log(M/(i+M))
            ytemp=np.array([self.data[i]])
            DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
            Stemp=np.array([0]).copy()
            Ntemp=np.array([1]).copy()
            logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
            
            normalisingConstant=logsumexp(logProbs)
            if newClusterInd==self.n+1: #observation is in a new cluster
                logDensity+=logProbs[K]-normalisingConstant
                K+=1
                Nprogressive[S[i]]=1
                
                
            else: #observation is in cluster newClusterInd=k
                logDensity+=logProbs[newClusterInd]-normalisingConstant
                Nprogressive[clusters[newClusterInd]]+=1
            
                
        return logDensity
    
    def GeyerEstimator(self,numIter,burnIn,distribution):
        if distribution=='prior':
            S1=[]
            N1=[]
            M1=[]
            S2=[]
            N2=[]
            M2=[]
            for i in range(numIter-burnIn):
                M1.append(self.priorSimM())
                out=self.priorAllocationSim(M1[-1])
                S1.append(out[0])
                N1.append(out[1])
            S2,N2,M2,_,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            M2=M2[burnIn:]
            
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            p11=np.zeros(numIter-burnIn)
            p21=np.zeros(numIter-burnIn)
            p12=np.zeros(numIter-burnIn)
            p22=np.zeros(numIter-burnIn)
            for i in range(numIter-burnIn):
                p11[i]=self.logPrior(S1[i],N1[i],M1[i])
                #p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])
                p12[i]=self.logPrior(S2[i],N2[i],M2[i])
                #p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])
            def objectiveLogLlk(eta2):
                p1=0
                p2=0
                for i in range(numIter-burnIn):
                    p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i],M1[i])+eta2
                    p1+=p11[i]-logsumexp(np.array([p11[i],p21[i]]))
                    
                    p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])+eta2
                    p2+=p22[i]-logsumexp(np.array([p12[i],p22[i]]))
                return(-(p1+p2))
            # def objectiveLogLlk(eta2):
            #     p1=0
            #     p2=0
            #     for i in range(numIter-burnIn):
            #         p11=self.logPrior(S1[i],N1[i])
            #         p21=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])+eta2
            #         p1+=p11-logsumexp(np.array([p11,p21]))
                    
            #         p12=self.logPrior(S2[i],N2[i])
            #         p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])+eta2
            #         p2+=p22-logsumexp(np.array([p12,p22]))
            #     return(-(p1+p2))
            res=minimize_scalar(objectiveLogLlk)
            return -res.x
        if distribution=='prior+Newton':
            S0=[]
            N0=[]
            S1=[]
            N1=[]
            S2=[]
            N2=[]
            
            ##Simulating the kind of approximate posterior distribution (like Newton algo) from Chib
            
            for t in range(numIter-burnIn):
                #np.random.shuffle(self.data) #shuffle data
                u=np.zeros(self.n)
                S=np.zeros(self.n,dtype=int)
                N=np.zeros(self.n,dtype=int)
                DPmixTemp=DPmixnorm1dKnownM(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                N[0]+=1
                clusters=np.array([0])
                K=1
                u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
                for i in range(1,self.n):
                    logProbs=np.zeros(K+1)
                    for k in range(K):
                        logProbs[k]+=np.log(N[clusters[k]]/(i+self.M))
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=clusters[k]#pretend y is in cluster k
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=self.n+1
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int)
                        Ntemp=np.array([DPmixtemp.n])
                        logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    logProbs[K]=logProbs[K]+np.log(self.M/(i+self.M))
                    ytemp=np.array([self.data[i]])
                    DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                    Stemp=np.array([0]).copy()
                    Ntemp=np.array([1]).copy()
                    logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    sampledCluster=catDistLogProb(logProbs)
                    if sampledCluster==K:
                        indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                        N[indexClusterAvailable]+=1
                        clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                        S[i]=indexClusterAvailable
                        K+=1
                        #print(K[t],indexClusterAvailable,N[t],S[t])
                    else:
                        #print('assigned to existing cluster',clusters[sampledCluster])
                        S[i]=clusters[sampledCluster]
                        N[clusters[sampledCluster]]+=1
                S0.append(S.copy())
                N0.append(N.copy())
                    
            #Simulating from prior
            for i in range(numIter-burnIn):
                out=self.priorAllocationSim()
                S1.append(out[0])
                N1.append(out[1])
            #Simulating from posterior
            S2,N2,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            p00=np.zeros(numIter-burnIn)
            p10=np.zeros(numIter-burnIn)
            #p20=np.zeros(numIter-burnIn)
            p01=np.zeros(numIter-burnIn)
            p11=np.zeros(numIter-burnIn)
            #p21=np.zeros(numIter-burnIn)
            p02=np.zeros(numIter-burnIn)
            p12=np.zeros(numIter-burnIn)
            #p22=np.zeros(numIter-burnIn)
            for i in range(numIter-burnIn):
                p00[i]=self.NewtonAllocationPosterior(S0[i], N0[i])
                p10[i]=self.logPrior(S0[i],N0[i])
                #p20[i]=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i])
                
                p01[i]=self.NewtonAllocationPosterior(S1[i], N1[i])
                p11[i]=self.logPrior(S1[i],N1[i])
                #p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])
                
                p02[i]=self.NewtonAllocationPosterior(S2[i], N2[i])
                p12[i]=self.logPrior(S2[i],N2[i])
                #p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])
                
            def objectiveLogLlk(eta2):
                p0=0
                p1=0
                p2=0
                for i in range(numIter-burnIn):
                    
                    p20=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i])+eta2
                    p0+=p00[i]-logsumexp(np.array([p00[i],p10[i],p20]))
                    
                    p21=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])+eta2
                    p1+=p11[i]-logsumexp(np.array([p01[i],p11[i],p21]))
                    
                    
                    p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])+eta2
                    p2+=p22-logsumexp(np.array([p02[i],p12[i],p22]))
                return(-(p0+p1+p2))
            res=minimize_scalar(objectiveLogLlk)
            return -res.x
        if distribution=='Newton':
            S0=[]
            N0=[]
            M0=[]

            
            S2=[]
            N2=[]
            M2=[]
            ##Simulating the kind of approximate posterior distribution (like Newton algo) from Chib
            
            for t in range(numIter-burnIn):
                #np.random.shuffle(self.data) #shuffle data
                M0.append(self.priorSimM())
                u=np.zeros(self.n)
                S=np.zeros(self.n,dtype=int)
                N=np.zeros(self.n,dtype=int)
                DPmixTemp=DPmixnorm1dLocation(data=self.data[0],priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                N[0]+=1
                clusters=np.array([0])
                K=1
                u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
                for i in range(1,self.n):
                    logProbs=np.zeros(K+1)
                    for k in range(K):
                        logProbs[k]+=np.log(N[clusters[k]]/(i+M0[-1]))
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=clusters[k]#pretend y is in cluster k
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=self.n+1
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int)
                        Ntemp=np.array([DPmixtemp.n])
                        logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    logProbs[K]=logProbs[K]+np.log(M0[-1]/(i+M0[-1]))
                    ytemp=np.array([self.data[i]])
                    DPmixtemp=DPmixnorm1dLocation(data=ytemp,priorDistLoc=self.priorDistLoc,priorParamLoc=self.priorParamLoc,sigma=self.sigma,priorDistM=self.priorDistM,priorParamM=self.priorParamM)
                    Stemp=np.array([0]).copy()
                    Ntemp=np.array([1]).copy()
                    logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    sampledCluster=catDistLogProb(logProbs)
                    if sampledCluster==K:
                        indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                        N[indexClusterAvailable]+=1
                        clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                        S[i]=indexClusterAvailable
                        K+=1
                        #print(K[t],indexClusterAvailable,N[t],S[t])
                    else:
                        #print('assigned to existing cluster',clusters[sampledCluster])
                        S[i]=clusters[sampledCluster]
                        N[clusters[sampledCluster]]+=1
                
                S0.append(S.copy())
                N0.append(N.copy())
                    
            
            #Simulating from posterior
            S2,N2,M2,_,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            M2=M2[burnIn:]
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            p00=np.zeros(numIter-burnIn)

            #p20=np.zeros(numIter-burnIn)
          
            
            p02=np.zeros(numIter-burnIn)
            #p22=np.zeros(numIter-burnIn)
            for i in range(numIter-burnIn):
                p00[i]=self.NewtonAllocationPosterior(S0[i], N0[i],M0[i])
                #p20[i]=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i])
                
          
                p02[i]=self.NewtonAllocationPosterior(S2[i], N2[i],M2[i])
                #p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])
                
            def objectiveLogLlk(eta2):
                p0=0
                p2=0
                for i in range(numIter-burnIn):
                    
                    p20=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i],M0[i])+eta2
                    p0+=p00[i]-logsumexp(np.array([p00[i],p20]))
                    
                  
                    
                    
                    p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i],M2[i])+eta2
                    p2+=p22-logsumexp(np.array([p02[i],p22]))
                return(-(p0+p2))
            res=minimize_scalar(objectiveLogLlk)
            return -res.x
            
class DPmixnorm1dKnownM:
    
    
    def __init__(self, data=None, priorDistLocScale=None,
                 priorParamLocScale=None, M=None):
        if data is None:
            sys.exit('please specify data')
        else:
            self.data=data
        
        if M is None:
            sys.exit('please specify a value for the concentration parameter of the Dirichlet Process')
        self.M=M
        self.n=data.shape[0]

        if priorDistLocScale is None:
            self.priorDistLocScale='Conjugate'
        else:
            self.priorDistLocScale=priorDistLocScale

        if priorParamLocScale is None:
            if self.priorDistLocScale=='Conjugate':
                self.priorParamLocScale=np.array([np.mean(self.data),2.6/(np.max(self.data)-np.min(self.data)),1.28,0.36*np.var(self.data)]) #b0, N0, c0, C0 according to p.178 Fruhwirth Schnatter, as chsen by raftery 1996
        else:
            self.priorParamLocScale=priorParamLocScale

        
   
    
    def logLikelihoodAllocations(self,S,N):
        clusters=np.unique(S)
        llk=0
        for k in range(len(clusters)):
            mu0=self.priorParamLocScale[0]
            nu=self.priorParamLocScale[1]
            alpha0=self.priorParamLocScale[2]
            beta0=self.priorParamLocScale[3]
            nk=N[clusters[k]]
            yk=self.data[S==clusters[k]]
            beta_n=beta0+0.5*nk*np.var(yk)+nk*nu/(2*(nu+nk))*(np.mean(yk)-mu0)**2
            llk+=-nk*0.5*np.log(2*np.pi*beta_n)+0.5*np.log(nu/(nu+nk))+alpha0*np.log(beta0/beta_n)+loggamma(alpha0+nk/2)-loggamma(alpha0)
        return llk
    def logPrior(self,S,N):
        clusters=np.unique(S)
        logPrior=0
        logPrior+=len(clusters)*np.log(self.M)+loggamma(self.M)-loggamma(self.M+self.n)
        for k in range(len(clusters)):
            logPrior+=loggamma(N[clusters[k]])
        return logPrior
    def GibbsSamplingClusters(self,numIter):
        S=[]
        N=[]
        K=np.zeros(numIter,dtype=int)
        
        #intialise from the prior
        S.append(np.zeros(self.n,dtype=int))
        N.append(np.zeros(self.n,dtype=int))
        S[0],N[0]=self.priorAllocationSim()
        clusters=np.unique(S[0])
        K[0]=len(clusters)
        
        for t in range(1,numIter):
            S.append(np.zeros(self.n,dtype=int))
            S[t]=S[t-1].copy()
            N.append(np.zeros(self.n,dtype=int))
            N[t]=N[t-1].copy()
            K[t]=K[t-1].copy()
            
            for i in range(self.n):
                
                #Remove y_i from the data
                NminusI=N[t].copy()
                NminusI[S[t][i]]=NminusI[S[t][i]]-1
                if NminusI[S[t][i]]==0:
                    #A cluster is being emptied
                    K[t]=K[t]-1
                    clusters=np.setdiff1d(clusters,np.array([S[t][i]]))
                logProbs=np.zeros(K[t]+1)
                for k in range(K[t]):
                    
                    logProbs[k]=logProbs[k]+np.log(NminusI[clusters[k]]/(self.n-1+self.M))
                    Stemp=S[t].copy()
                    Stemp[i]=clusters[k]#pretend y is in cluster k
                    ytemp=self.data[Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[t].copy()
                    Stemp[i]=self.n+1
                    ytemp=self.data[Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int)
                    Ntemp=np.array([DPmixtemp.n])
                    logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                 
                    
                
                logProbs[K[t]]=logProbs[K[t]]+np.log(self.M/(self.n-1+self.M))
                ytemp=np.array([self.data[i]])
                DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                
                logProbs[K[t]]=logProbs[K[t]]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)

                sampledCluster=catDistLogProb(logProbs)
                
                if sampledCluster==K[t]:
                    indexClusterAvailable=next((idx for idx, val in np.ndenumerate(NminusI) if val==0))[0] #Find the first zero cluster
                    NminusI[indexClusterAvailable]+=1
                    clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                    S[t][i]=indexClusterAvailable
                    N[t]=NminusI.copy()
                    K[t]+=1
                    #print(K[t],indexClusterAvailable,N[t],S[t])
                else:
                    #print('assigned to existing cluster',clusters[sampledCluster])
                    S[t][i]=clusters[sampledCluster]
                    NminusI[clusters[sampledCluster]]+=1
                    N[t]=NminusI.copy()
                
        
        return(S,N,K)
    def ChibEstimator(self,numIter):
        #Likelihood estimation
        w=np.zeros(numIter)
        for t in range(numIter):
            np.random.shuffle(self.data) #shuffle data
            u=np.zeros(self.n)
            S=np.zeros(self.n,dtype=int)
            N=np.zeros(self.n,dtype=int)
            DPmixTemp=DPmixnorm1dKnownM(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
            N[0]+=1
            clusters=np.array([0])
            K=1
            u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
            for i in range(1,self.n):
                logProbs=np.zeros(K+1)
                for k in range(K):
                    logProbs[k]+=np.log(N[clusters[k]]/(i+self.M))
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=clusters[k]#pretend y is in cluster k
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                    Ntemp=np.array([DPmixtemp.n]).copy()
                    logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=self.n+1
                    ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                    DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                    Stemp=np.zeros(DPmixtemp.n,dtype=int)
                    Ntemp=np.array([DPmixtemp.n])
                    logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                logProbs[K]=logProbs[K]+np.log(self.M/(i+self.M))
                ytemp=np.array([self.data[i]])
                DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                Stemp=np.array([0]).copy()
                Ntemp=np.array([1]).copy()
                logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                sampledCluster=catDistLogProb(logProbs)
                if sampledCluster==K:
                    indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                    N[indexClusterAvailable]+=1
                    clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                    S[i]=indexClusterAvailable
                    K+=1
                    #print(K[t],indexClusterAvailable,N[t],S[t])
                else:
                    #print('assigned to existing cluster',clusters[sampledCluster])
                    S[i]=clusters[sampledCluster]
                    N[clusters[sampledCluster]]+=1
                #print(i,clusters,K)
                
                u[i]=logsumexp(logProbs)
            w[t]=np.sum(u)
        llk=-np.log(numIter)+logsumexp(w)
        print('llk uncertainty sd', 1/np.sqrt(numIter)*np.sqrt(np.var(np.exp(w)))/np.mean(np.exp(w)))
        print('llk', llk)
        return llk
    
    def arithmeticMean(self,numSim):
        llk=np.zeros(numSim)
        for i in range(numSim):
            S,N=self.priorAllocationSim()
            llk[i]=self.logLikelihoodAllocations(S, N)
        return -np.log(numSim)+logsumexp(llk)
    
    def priorAllocationSim(self):
        S=np.zeros(self.n, dtype='int')
        N=np.zeros(self.n, dtype='int')
        N[0]=1
        for i in range(1,self.n):
            if self.M/(self.M+i)>np.random.random():
                S[i]=np.max(S)+1
                N[S[i]]+=1
            else:
                newIndex=np.random.choice(self.n,p=N/i)
                S[i]=newIndex
                N[newIndex]+=1
        return(S,N)
    def NewtonAllocationPosterior(self,S,N):
        
        logDensity=0
        K=1
        #np.unique without sorting
        indexes = np.unique(S, return_index=True)[1]
        clusters=[S[index] for index in sorted(indexes)]
        Nprogressive=np.zeros(self.n,dtype=int)-1
        Nprogressive[S[0]]=1
       
        for i in range(1,self.n):
            logProbs=np.zeros(K+1)
            newClusterInd=self.n+1
            
            for k in range(K):
                if S[i]==clusters[k]:
                    newClusterInd=k
                logProbs[k]+=np.log(Nprogressive[clusters[k]]/(i+self.M))
                Stemp=S[:(i+1)].copy()
                Stemp[i]=clusters[k]#pretend y is in cluster k
                ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                Ntemp=np.array([DPmixtemp.n]).copy()
                logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                
                Stemp=S[:(i+1)].copy()
                Stemp[i]=self.n+1
                ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                Stemp=np.zeros(DPmixtemp.n,dtype=int)
                Ntemp=np.array([DPmixtemp.n])
                logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                
            logProbs[K]=logProbs[K]+np.log(self.M/(i+self.M))
            ytemp=np.array([self.data[i]])
            DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
            Stemp=np.array([0]).copy()
            Ntemp=np.array([1]).copy()
            logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
            
            normalisingConstant=logsumexp(logProbs)
            if newClusterInd==self.n+1: #observation is in a new cluster
                logDensity+=logProbs[K]-normalisingConstant
                K+=1
                Nprogressive[S[i]]=1
                
                
            else: #observation is in cluster newClusterInd=k
                logDensity+=logProbs[newClusterInd]-normalisingConstant
                Nprogressive[clusters[newClusterInd]]+=1
            
                
        return logDensity
    def GeyerEstimator(self,numIter,burnIn,distribution):
        if distribution=='prior':
            S1=[]
            N1=[]
            S2=[]
            N2=[]
            for i in range(numIter-burnIn):
                out=self.priorAllocationSim()
                S1.append(out[0])
                N1.append(out[1])
            S2,N2,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            p11=np.zeros(numIter-burnIn)
            p21=np.zeros(numIter-burnIn)
            p12=np.zeros(numIter-burnIn)
            p22=np.zeros(numIter-burnIn)
            for i in range(numIter-burnIn):
                p11[i]=self.logPrior(S1[i],N1[i])
                #p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])
                p12[i]=self.logPrior(S2[i],N2[i])
                #p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])
            def objectiveLogLlk(eta2):
                p1=0
                p2=0
                for i in range(numIter-burnIn):
                    p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])+eta2
                    p1+=p11[i]-logsumexp(np.array([p11[i],p21[i]]))
                    
                    p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])+eta2
                    p2+=p22[i]-logsumexp(np.array([p12[i],p22[i]]))
                return(-(p1+p2))
            # def objectiveLogLlk(eta2):
            #     p1=0
            #     p2=0
            #     for i in range(numIter-burnIn):
            #         p11=self.logPrior(S1[i],N1[i])
            #         p21=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])+eta2
            #         p1+=p11-logsumexp(np.array([p11,p21]))
                    
            #         p12=self.logPrior(S2[i],N2[i])
            #         p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])+eta2
            #         p2+=p22-logsumexp(np.array([p12,p22]))
            #     return(-(p1+p2))
            res=minimize_scalar(objectiveLogLlk)
            return -res.x
        if distribution=='prior+Newton':
            S0=[]
            N0=[]
            S1=[]
            N1=[]
            S2=[]
            N2=[]
            
            ##Simulating the kind of approximate posterior distribution (like Newton algo) from Chib
            
            for t in range(numIter-burnIn):
                #np.random.shuffle(self.data) #shuffle data
                u=np.zeros(self.n)
                S=np.zeros(self.n,dtype=int)
                N=np.zeros(self.n,dtype=int)
                DPmixTemp=DPmixnorm1dKnownM(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                N[0]+=1
                clusters=np.array([0])
                K=1
                u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
                for i in range(1,self.n):
                    logProbs=np.zeros(K+1)
                    for k in range(K):
                        logProbs[k]+=np.log(N[clusters[k]]/(i+self.M))
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=clusters[k]#pretend y is in cluster k
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=self.n+1
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int)
                        Ntemp=np.array([DPmixtemp.n])
                        logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    logProbs[K]=logProbs[K]+np.log(self.M/(i+self.M))
                    ytemp=np.array([self.data[i]])
                    DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                    Stemp=np.array([0]).copy()
                    Ntemp=np.array([1]).copy()
                    logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    sampledCluster=catDistLogProb(logProbs)
                    if sampledCluster==K:
                        indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                        N[indexClusterAvailable]+=1
                        clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                        S[i]=indexClusterAvailable
                        K+=1
                        #print(K[t],indexClusterAvailable,N[t],S[t])
                    else:
                        #print('assigned to existing cluster',clusters[sampledCluster])
                        S[i]=clusters[sampledCluster]
                        N[clusters[sampledCluster]]+=1
                S0.append(S.copy())
                N0.append(N.copy())
                    
            #Simulating from prior
            for i in range(numIter-burnIn):
                out=self.priorAllocationSim()
                S1.append(out[0])
                N1.append(out[1])
            #Simulating from posterior
            S2,N2,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            p00=np.zeros(numIter-burnIn)
            p10=np.zeros(numIter-burnIn)
            #p20=np.zeros(numIter-burnIn)
            p01=np.zeros(numIter-burnIn)
            p11=np.zeros(numIter-burnIn)
            #p21=np.zeros(numIter-burnIn)
            p02=np.zeros(numIter-burnIn)
            p12=np.zeros(numIter-burnIn)
            #p22=np.zeros(numIter-burnIn)
            for i in range(numIter-burnIn):
                p00[i]=self.NewtonAllocationPosterior(S0[i], N0[i])
                p10[i]=self.logPrior(S0[i],N0[i])
                #p20[i]=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i])
                
                p01[i]=self.NewtonAllocationPosterior(S1[i], N1[i])
                p11[i]=self.logPrior(S1[i],N1[i])
                #p21[i]=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])
                
                p02[i]=self.NewtonAllocationPosterior(S2[i], N2[i])
                p12[i]=self.logPrior(S2[i],N2[i])
                #p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])
                
            def objectiveLogLlk(eta2):
                p0=0
                p1=0
                p2=0
                for i in range(numIter-burnIn):
                    
                    p20=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i])+eta2
                    p0+=p00[i]-logsumexp(np.array([p00[i],p10[i],p20]))
                    
                    p21=self.logLikelihoodAllocations(S1[i], N1[i])+self.logPrior(S1[i], N1[i])+eta2
                    p1+=p11[i]-logsumexp(np.array([p01[i],p11[i],p21]))
                    
                    
                    p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])+eta2
                    p2+=p22-logsumexp(np.array([p02[i],p12[i],p22]))
                return(-(p0+p1+p2))
            res=minimize_scalar(objectiveLogLlk)
            return -res.x
        if distribution=='Newton':
            S0=[]
            N0=[]

            
            S2=[]
            N2=[]
            
            ##Simulating the kind of approximate posterior distribution (like Newton algo) from Chib
            
            for t in range(numIter-burnIn):
                #np.random.shuffle(self.data) #shuffle data
                u=np.zeros(self.n)
                S=np.zeros(self.n,dtype=int)
                N=np.zeros(self.n,dtype=int)
                DPmixTemp=DPmixnorm1dKnownM(data=self.data[0],priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                N[0]+=1
                clusters=np.array([0])
                K=1
                u[0]=DPmixTemp.logLikelihoodAllocations(np.array([0]),np.array([1]))
                for i in range(1,self.n):
                    logProbs=np.zeros(K+1)
                    for k in range(K):
                        logProbs[k]+=np.log(N[clusters[k]]/(i+self.M))
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=clusters[k]#pretend y is in cluster k
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int).copy()
                        Ntemp=np.array([DPmixtemp.n]).copy()
                        logProbs[k]=logProbs[k]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                        
                        Stemp=S[:(i+1)].copy()
                        Stemp[i]=self.n+1
                        ytemp=self.data[:(i+1)][Stemp==clusters[k]]
                        DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                        Stemp=np.zeros(DPmixtemp.n,dtype=int)
                        Ntemp=np.array([DPmixtemp.n])
                        logProbs[k]=logProbs[k]-DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    logProbs[K]=logProbs[K]+np.log(self.M/(i+self.M))
                    ytemp=np.array([self.data[i]])
                    DPmixtemp=DPmixnorm1dKnownM(data=ytemp,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale,M=self.M)
                    Stemp=np.array([0]).copy()
                    Ntemp=np.array([1]).copy()
                    logProbs[K]=logProbs[K]+DPmixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    sampledCluster=catDistLogProb(logProbs)
                    if sampledCluster==K:
                        indexClusterAvailable=np.max(S[:(i+1)])+1 #Find the first zero cluster
                        N[indexClusterAvailable]+=1
                        clusters=np.insert(clusters,indexClusterAvailable,indexClusterAvailable)
                        S[i]=indexClusterAvailable
                        K+=1
                        #print(K[t],indexClusterAvailable,N[t],S[t])
                    else:
                        #print('assigned to existing cluster',clusters[sampledCluster])
                        S[i]=clusters[sampledCluster]
                        N[clusters[sampledCluster]]+=1
                S0.append(S.copy())
                N0.append(N.copy())
                    
            
            #Simulating from posterior
            S2,N2,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            p00=np.zeros(numIter-burnIn)

            #p20=np.zeros(numIter-burnIn)
          
            
            p02=np.zeros(numIter-burnIn)
            #p22=np.zeros(numIter-burnIn)
            for i in range(numIter-burnIn):
                p00[i]=self.NewtonAllocationPosterior(S0[i], N0[i])
                #p20[i]=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i])
                
          
                p02[i]=self.NewtonAllocationPosterior(S2[i], N2[i])
                #p22[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])
                
            def objectiveLogLlk(eta2):
                p0=0
                p2=0
                for i in range(numIter-burnIn):
                    
                    p20=self.logLikelihoodAllocations(S0[i], N0[i])+self.logPrior(S0[i], N0[i])+eta2
                    p0+=p00[i]-logsumexp(np.array([p00[i],p20]))
                    
                  
                    
                    
                    p22=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])+eta2
                    p2+=p22-logsumexp(np.array([p02[i],p22]))
                return(-(p0+p2))
            res=minimize_scalar(objectiveLogLlk)
            return -res.x
    def GeyerEstimatorSKlearn(self,numIter,burnIn,distribution):
        if distribution=='prior':
            S1=[]
            N1=[]
            S2=[]
            N2=[]
            logH1=np.zeros(numIter-burnIn)
            logH2=np.zeros(numIter-burnIn)
            
            S2,N2,_=self.GibbsSamplingClusters(numIter)
            S2=S2[burnIn:]
            N2=N2[burnIn:]
            for i in range(numIter-burnIn):
                out=self.priorAllocationSim()
                S1.append(out[0])
                N1.append(out[1])
                logH1[i]=self.logPrior(S1[i], N1[i])
                
                logH2[i]=self.logLikelihoodAllocations(S2[i], N2[i])+self.logPrior(S2[i], N2[i])
                
            response=np.concatenate([np.ones(numIter-burnIn),np.zeros(numIter-burnIn)])
            predictor=np.concatenate([logH1,logH2])
            #predictor=np.concatenate([logH1,logH2]).reshape(-1, 1)
            #clf = LogisticRegression().fit(predictor, response)
            #print(clf.intercept_,clf.coef_)
            ######## WATCH OUT : only works in this formulation for equal sample sizes. Otherwise modify eta accordingly #######
            df = pd.DataFrame({'response':response, 'predictor':predictor})
            print(predictor)
            res1=smf.glm('response ~ 1', data=df,offset=predictor,family=sm.families.Binomial()).fit()
            print(res1.summary())
            return res1
 