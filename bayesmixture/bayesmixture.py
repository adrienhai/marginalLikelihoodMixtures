#ATTENTION AU DPM : ESS a l'air faux dans SMC puis le resamplig aussi

import sys
import numpy as np
import scipy.stats as ss
import math
import time
from itertools import permutations
from scipy.special import logsumexp
from scipy.special import loggamma

def catDistLogProb(logProb):
    n=len(logProb)

    #sim n Gumbel RVs
    g=-np.log(-np.log(np.random.random(n)))
    return np.argmax(g+logProb)

def log_domain_mean(logx):
        "np.log(np.mean(np.exp(x))) but more stable"
        n = len(logx)
        damax = np.max(logx)
        return np.log(np.sum(np.exp(logx-damax))) \
        + damax-np.log(n)
def log_domain_var(logx):
        """like np.log(np.var(np.exp(logx)))
        except more stable"""
        n = len(logx)
        log_xmean = log_domain_mean(logx)
        return np.log(np.sum( np.expm1(logx-log_xmean)**2))\
        + 2*log_xmean - np.log(n)
def canonical_form_partition(li):
    out=np.zeros(len(li),dtype=int)  
    first = dict()
    for i in range(len(li)):
        v = first.get(li[i])
        if v is None:
            first[li[i]] = i
            v = i
        out[i] = v
    return out

class mixnorm1d :

    def __init__(self, data=None, k=None, priorDistWeight=None,priorParamWeight=None,priorDistLocScale=None,
                 priorParamLocScale=None):
        if data is None:
            sys.exit('please specify data')
        else:
            self.data=data

        if k is None:
            sys.exit('please specify k, the number of mixture components')
        else:
            self.k=k

        self.n=data.shape[0]

        if priorDistWeight is None:
            self.priorDistWeight='Dirichlet'
        else:
            self.priorDistWeight=priorDistWeight

        if priorParamWeight is None:
            self.priorParamWeight=np.ones(k)
        else:
            self.priorParamWeight=priorParamWeight

        if priorDistLocScale is None:
            self.priorDistLocScale='ConditionalConjugate'
        else:
            self.priorDistLocScale=priorDistLocScale

        if priorParamLocScale is None:
            self.priorParamLocScale=np.array([np.mean(self.data),2.6/(np.max(self.data)-np.min(self.data)),1.28,0.36*np.var(self.data)]) #b0, N0, c0, C0 according to p.178 Fruhwirth Schnatter, as chsen by
        else:
            self.priorParamLocScale=priorParamLocScale
        
    def logLikelihood(self,paramWeight,paramLocScale): #Attention ça a l'air faux
        if paramWeight.shape[1]!=self.k:
            sys.exit('The number of weights is different from the number of components.')
        llk=0
        for i in range(self.k):
            llk+=paramWeight[0][i]*ss.norm.pdf(self.data,paramLocScale[i,0],np.sqrt(paramLocScale[i,1]))
        llk=np.sum(np.log(llk))
        return llk
    
    def logLikelihoodV2(self,paramWeight,paramLoc,paramScale): #Attention ça a l'air faux
        llk=0
        for i in range(self.n):
            indLlk=np.zeros(self.k)
            for k in range(self.k):
                indLlk[k]=np.log(paramWeight[k])+ss.norm.logpdf(self.data[i],paramLoc[k],np.sqrt(paramScale[k]))
            llk+=logsumexp(indLlk)
        return llk
    
    def augmentedLogLikelihood(self, paramWeight, paramLocScale, paramAllocation, paramCount):
        if paramWeight.shape[1]!=self.k:
            sys.exit('The number of weights is different from the number of components.')
        llk=0
        for i in range(self.k):
            llk+=paramCount[i]*np.log(paramWeight[0,i])+np.sum(ss.norm.logpdf(self.data[paramAllocation==i], loc=paramLocScale[i,0], scale=np.sqrt(paramLocScale[i,1])))
        return llk
    
    def logEPPF(self,N): #https://mimno.infosci.cornell.edu/info6150/exercises/polya.pdf
        id_non_empty_clusters=(N>0)
        
        K_plus=np.sum(id_non_empty_clusters)
        out=loggamma(np.sum(self.priorParamWeight))-loggamma(np.sum(self.priorParamWeight)+self.n)+np.log(math.factorial(self.k))-np.log(math.factorial(self.k-K_plus))
        for k in range(self.k):
            out+=loggamma(self.priorParamWeight[k]+N[k])-loggamma(self.priorParamWeight[k])
        return out

    def logPriorAllocation(self,S,N):
        
        logPrior=loggamma(np.sum(self.priorParamWeight))-loggamma(self.n+np.sum(self.priorParamWeight))
        for k in range(self.k):
            logPrior+=loggamma(N[k]+self.priorParamWeight[k])-loggamma(self.priorParamWeight[k])
        return logPrior
    def logLikelihoodAllocations(self,S,N):
        if self.priorDistLocScale=='ConditionalConjugate':
            llk=0
            for k in range(self.k):
                if N[k]!=0:
                    mu0=self.priorParamLocScale[0]
                    nu=self.priorParamLocScale[1]
                    alpha0=self.priorParamLocScale[2]
                    beta0=self.priorParamLocScale[3]
                    nk=N[k]
                    yk=self.data[S==k]
                    beta_n=beta0+0.5*nk*np.var(yk)+nk*nu/(2*(nu+nk))*(np.mean(yk)-mu0)**2
                    llk+=-nk*0.5*np.log(2*np.pi*beta_n)+0.5*np.log(nu/(nu+nk))+alpha0*np.log(beta0/beta_n)+loggamma(alpha0+nk/2)-loggamma(alpha0)
            return llk
        else:
            sys.exit('Model must be conjugate')

            
   
    def augmentedLogPosterior(self, paramWeight, paramLocScale, S, N):

        post=ss.dirichlet.logpdf(paramWeight[0,:],alpha=self.priorParamWeight+N)
        c=np.zeros(self.k)
        C=np.zeros(self.k)
        b=np.zeros(self.k)
        B=np.zeros(self.k)
        for i in range(self.k):
            if N[i]==0: #empty cluster
                group_mean=0
                group_var=0
            else:
                group_mean=np.mean(self.data[S==i])
                group_var=np.var(self.data[S==i])

            #shape of invgamma
            c[i]=self.priorParamLocScale[2]+0.5*N[i] #F-S eq (6.15)

            #scale of invgamma
            C[i]=self.priorParamLocScale[3]+0.5*(N[i]*group_var+(N[i]*self.priorParamLocScale[1])/(N[i]+self.priorParamLocScale[1])*(group_mean-self.priorParamLocScale[0])*(group_mean-self.priorParamLocScale[0]))

            #shape of normal dist
            B[i]=paramLocScale[i,1]/(N[i]+self.priorParamLocScale[1])

            #loc of normal dist
            b[i]=self.priorParamLocScale[1]*self.priorParamLocScale[0]/(N[i]+self.priorParamLocScale[1])+N[i]*group_mean/(N[i]+self.priorParamLocScale[1])

            #add to post
        post+=np.sum(ss.norm.logpdf(paramLocScale[:,0],loc=b,scale=np.sqrt(B)))+np.sum(ss.invgamma.logpdf(paramLocScale[:,1],a=c,scale=C))
        return post

    def augmentedLogPosteriorV2(self, paramWeight, paramLoc,paramScale, S, N):
        
        post=ss.dirichlet.logpdf(paramWeight,alpha=self.priorParamWeight+N)
        
        c=np.zeros(self.k)
        C=np.zeros(self.k)
        b=np.zeros(self.k)
        B=np.zeros(self.k)
        for i in range(self.k):
            if N[i]==0: #empty cluster
                group_mean=0
                group_var=0
            else:
                group_mean=np.mean(self.data[S==i])
                group_var=np.var(self.data[S==i])

            #shape of invgamma
            c[i]=self.priorParamLocScale[2]+0.5*N[i] #F-S eq (6.15)

            #scale of invgamma
            C[i]=self.priorParamLocScale[3]+0.5*(N[i]*group_var+((N[i]*self.priorParamLocScale[1])/(N[i]+self.priorParamLocScale[1]))*(group_mean-self.priorParamLocScale[0])**2)

            #shape of normal dist
            B[i]=paramScale[i]/(N[i]+self.priorParamLocScale[1])

            #loc of normal dist
            b[i]=self.priorParamLocScale[1]*self.priorParamLocScale[0]/(N[i]+self.priorParamLocScale[1])+N[i]*group_mean/(N[i]+self.priorParamLocScale[1])

            #add to post
            post+=ss.norm.logpdf(paramLoc[i],loc=b[i],scale=np.sqrt(B[i]))+ss.invgamma.logpdf(paramScale[i],a=c[i],scale=C[i])
        return post

    def logPriorWeight(self,paramWeight):
        return ss.dirichlet.logpdf(paramWeight[0,:],alpha=self.priorParamWeight)
    def logPriorWeightV2(self,paramWeight):
        return ss.dirichlet.logpdf(paramWeight,alpha=self.priorParamWeight)
    def logPriorLocScale(self,paramLocScale):
        out=0
        for i in range(self.k):
            out+=ss.norm.logpdf(paramLocScale[i,0],loc=self.priorParamLocScale[0],scale=np.sqrt(paramLocScale[i,1]/self.priorParamLocScale[1]))
            out+=ss.invgamma.logpdf(paramLocScale[i,1],a=self.priorParamLocScale[2],scale=self.priorParamLocScale[3])
        return out

    def logPriorScale(self,paramScale):
        return np.sum(ss.invgamma.logpdf(paramScale,a=self.priorParamLocScale[2],scale=self.priorParamLocScale[3]))

    def logPriorLoc(self,paramLoc,paramScale):
        if self.priorDistLocScale=='ConditionalConjugate':
            return np.sum(ss.norm.logpdf(paramLoc,loc=self.priorParamLocScale[0],scale=np.sqrt(paramScale/self.priorParamLocScale[1])))
        elif self.priorDistLocScale=='IndependencePrior':
            return np.sum(ss.norm.logpdf(paramLoc,loc=self.priorParamLocScale[0],scale=np.sqrt(self.priorParamLocScale[1])))
    def priorSimWeight(self):
        if self.priorDistWeight=='Dirichlet':
            return ss.dirichlet.rvs(alpha=self.priorParamWeight)
        else:
            sys.exit('non-supported prior distribution on the mixture weights')

    def priorSimScale(self):
        return ss.invgamma.rvs(a=self.priorParamLocScale[2],scale=self.priorParamLocScale[3],size=self.k)
    def priorSimLoc(self, paramScale):
        return ss.norm.rvs(loc=self.priorParamLocScale[0],scale=np.sqrt(paramScale/self.priorParamLocScale[1]))
    
    def logPosteriorPartition(self,S,N): #p.66 F-S mixtures
        return self.logEPPF(N)+self.logLikelihoodAllocations(S, N)
    def GibbsSamplerPartition(self,numIter,burnIn): ###############WARNING : s'occuper d'effectivement supprimer le burnIN
        S=[]
        N=[]
        S.append(np.zeros(self.n,dtype=int))
        N.append(np.zeros(self.k,dtype=int))
        N[-1][0]=self.n
        for t in range(numIter):
            print(t/numIter*100,end="\r")

            for i in range(self.n):
                
                #Remove y_i from the data
                NminusI=N[t].copy()
                NminusI[S[t][i]]=NminusI[S[t][i]]-1

                logProbs=np.zeros(self.k)
                for k in range(self.k):
                    
                    Stemp=S[t].copy()
                    Ntemp=NminusI.copy()
                    #pretend y is in cluster k
                    Ntemp[k]+=1
                    Stemp[i]=k
                    #ytemp=self.data[Stemp==k]
                    #Mixtemp=mixnorm1d(data=ytemp, k=self.k, priorDistWeight=self.priorDistWeight,priorParamWeight=self.priorParamWeight,priorDistLocScale=self.priorDistLocScale,
                 #priorParamLocScale=self.priorParamLocScale)
                
                    logProbs[k]=logProbs[k]+self.logLikelihoodAllocations(Stemp,Ntemp)+self.logPriorAllocation(Stemp, Ntemp)


                sampledCluster=catDistLogProb(logProbs)
                
                NminusI[sampledCluster]+=1
                S[t][i]=sampledCluster
                N[t]=NminusI.copy()
                
            S.append(S[t].copy())
            N.append(N[t].copy())
        
        return(S,N)    
            
            
            
    def GibbsSamplerV2(self,numIter,burnIn): #algo 3.5 F-S
        if self.priorDistLocScale=='ConditionalConjugate':
            S=[] #vector of allocation of the observation to a cluster
            N=[] #vector of counts of the number of observations in each cluster
            sigma2=[] #scale mixture params
            mu=[] #loc mixture params
            eta=[] #weights
            
            S.append(np.zeros(self.n,dtype=int))
            N.append(np.zeros(self.k,dtype=int))
            N[-1][0]=self.n
            for t in range(numIter):
                
                #sample the weights
                eta.append(ss.dirichlet.rvs(alpha=self.priorParamWeight+N[t])[0])
                #sample the scale and loc parameters for each k group
                c=np.zeros(self.k)
                C=np.zeros(self.k)
                b=np.zeros(self.k)
                B=np.zeros(self.k)
                sig_array=np.zeros(self.k)
                mu_array=np.zeros(self.k)
    
                for i in range(self.k):
    
                    if N[t][i]==0: #empty cluster
                        group_mean=0
                        group_var=0
                    else:
                        group_mean=np.mean(self.data[S[t]==i])
                        group_var=np.var(self.data[S[t]==i])
    
                    #shape of invgamma
                    c[i]=self.priorParamLocScale[2]+0.5*N[t][i] #F-S eq (6.15)
    
                    #scale of invgamma
                    C[i]=self.priorParamLocScale[3]+0.5*(N[t][i]*group_var+(N[t][i]*self.priorParamLocScale[1])/(N[t][i]+self.priorParamLocScale[1])*(group_mean-self.priorParamLocScale[0])*(group_mean-self.priorParamLocScale[0]))
    
                    #sample sigma2
                    sig_array[i]=ss.invgamma.rvs(a=c[i],scale=C[i])
    
                    #shape of normal dist
                    B[i]=sig_array[i]/(N[t][i]+self.priorParamLocScale[1])
    
                    #loc of normal dist
                    b[i]=self.priorParamLocScale[1]*self.priorParamLocScale[0]/(N[t][i]+self.priorParamLocScale[1])+N[t][i]*group_mean/(N[t][i]+self.priorParamLocScale[1])
    
                    #sample mu
                    mu_array[i]=ss.norm.rvs(loc=b[i],scale=np.sqrt(B[i]))
                    
                sigma2.append(sig_array)
                mu.append(mu_array)
    
                #Update allocations and counts for each observation
                S_array=np.zeros(self.n,dtype=int)
                N_array=np.zeros(self.k,dtype=int)
    
                weights_dist_S=np.zeros((self.n,self.k))
                for k in range(self.k):
                    weights_dist_S[:,k]=np.log(eta[t][k])+ss.norm.logpdf(self.data,loc=mu_array[k], scale=np.sqrt(sig_array[k]))[:,0]
                for i in range(self.n):
                    S_array[i]=np.int(catDistLogProb(weights_dist_S[i,:]))
                    N_array[np.int(S_array[i])]+=1
                S.append(S_array.copy())
                N.append(N_array.copy())
                
                # permut=np.random.permutation(self.k)
                # S[-1]=permut[S[-1]]
                # eta[-1]=eta[-1][permut]
                # mu[-1]=mu[-1][permut]
                # sigma2[-1]=sigma2[-1][permut]
                #[-1]=S[-1][permut]
            del S[0:(burnIn+1)]
            del N[0:(burnIn+1)]
            del eta[0:burnIn]
            del mu[0:burnIn]
            del sigma2[0:burnIn]
            return(eta,mu,sigma2,S,N)    

    def GibbsSamplerStep(self,S,N,eta,mu,sigma2):
        
        eta=ss.dirichlet.rvs(alpha=self.priorParamWeight+N)[0]
        c=np.zeros(self.k)
        C=np.zeros(self.k)
        b=np.zeros(self.k)
        B=np.zeros(self.k)
        sig_array=np.zeros(self.k)
        mu_array=np.zeros(self.k)
    
        for i in range(self.k):
    
            if N[i]==0: #empty cluster
                group_mean=0
                group_var=0
            else:
                group_mean=np.mean(self.data[S==i])
                group_var=np.var(self.data[S==i])

            #shape of invgamma
            c[i]=self.priorParamLocScale[2]+0.5*N[i] #F-S eq (6.15)

            #scale of invgamma
            C[i]=self.priorParamLocScale[3]+0.5*(N[i]*group_var+(N[i]*self.priorParamLocScale[1])/(N[i]+self.priorParamLocScale[1])*(group_mean-self.priorParamLocScale[0])*(group_mean-self.priorParamLocScale[0]))

            #sample sigma2
            sig_array[i]=ss.invgamma.rvs(a=c[i],scale=C[i])

            #shape of normal dist
            B[i]=sig_array[i]/(N[i]+self.priorParamLocScale[1])

            #loc of normal dist
            b[i]=self.priorParamLocScale[1]*self.priorParamLocScale[0]/(N[i]+self.priorParamLocScale[1])+N[i]*group_mean/(N[i]+self.priorParamLocScale[1])

            #sample mu
            mu_array[i]=ss.norm.rvs(loc=b[i],scale=np.sqrt(B[i]))
            
        

        #Update allocations and counts for each observation
        S_array=np.zeros(self.n,dtype=int)
        N_array=np.zeros(self.k,dtype=int)

        weights_dist_S=np.zeros((self.n,self.k))
        for k in range(self.k):
            weights_dist_S[:,k]=np.log(eta[k])+ss.norm.logpdf(self.data,loc=mu_array[k], scale=np.sqrt(sig_array[k]))[:,0]
        for i in range(self.n):
            S_array[i]=np.int(catDistLogProb(weights_dist_S[i,:]))
            N_array[np.int(S_array[i])]+=1
        
        
        # permut=np.random.permutation(self.k)
        # S[-1]=permut[S[-1]]
        # eta[-1]=eta[-1][permut]
        # mu[-1]=mu[-1][permut]
        # sigma2[-1]=sigma2[-1][permut]
        #[-1]=S[-1][permut]
        return(eta,mu_array,sig_array,S_array,N_array)    
                
    def GibbsSampler(self, numIter,burnIn,verbose=False):
        
        if self.priorDistLocScale=='ConditionalConjugate':
            S=[] #vector of allocation of the observation to a cluster
            N=[] #vector of counts of the number of observations in each cluster
            sigma2=[] #scale mixture params
            mu=[] #loc mixture params
            eta=[] #weights
            S.append(np.resize(np.arange(self.k),self.n)) #assign a cluster to each observation S[0] has shape (n,)
            N.append(np.zeros(self.k))
            for i in range(self.k):
                N[0][i]=np.sum(S[0]==i) #N[0] counts the number of obs in each k clusters
    
            for t in range(numIter): #Warning : N and S are one index late compared to eta and Theta
                if verbose==True:
                    print(t/numIter*100,end="\r")
    
    
                #sample the weights
                eta.append(ss.dirichlet.rvs(alpha=self.priorParamWeight+N[t],size=1))
                #sample the scale and loc parameters for each k group
                c=np.zeros(self.k)
                C=np.zeros(self.k)
                b=np.zeros(self.k)
                B=np.zeros(self.k)
                sig_array=np.zeros(self.k)
                mu_array=np.zeros(self.k)
    
                for i in range(self.k):
    
                    if N[t][i]==0: #empty cluster
                        group_mean=0
                        group_var=0
                    else:
                        group_mean=np.mean(self.data[S[t]==i])
                        group_var=np.var(self.data[S[t]==i])
    
                    #shape of invgamma
                    c[i]=self.priorParamLocScale[2]+0.5*N[t][i] #F-S eq (6.15)
    
                    #scale of invgamma
                    C[i]=self.priorParamLocScale[3]+0.5*(N[t][i]*group_var+(N[t][i]*self.priorParamLocScale[1])/(N[t][i]+self.priorParamLocScale[1])*(group_mean-self.priorParamLocScale[0])*(group_mean-self.priorParamLocScale[0]))
    
                    #sample sigma2
                    sig_array[i]=ss.invgamma.rvs(a=c[i],scale=C[i],size=1)
    
                    #shape of normal dist
                    B[i]=sig_array[i]/(N[t][i]+self.priorParamLocScale[1])
    
                    #loc of normal dist
                    b[i]=self.priorParamLocScale[1]*self.priorParamLocScale[0]/(N[t][i]+self.priorParamLocScale[1])+N[t][i]*group_mean/(N[t][i]+self.priorParamLocScale[1])
    
                    #sample mu
                    mu_array[i]=ss.norm.rvs(loc=b[i],scale=np.sqrt(B[i]),size=1)
    
    
                sigma2.append(sig_array)
                mu.append(mu_array)
    
                #Update allocations and counts for each observation
                S_array=np.zeros(self.n)
                N_array=np.zeros(self.k)
    
                weights_dist_S=np.zeros((self.n,self.k))
                for k in range(self.k):
                    weights_dist_S[:,k]=eta[t][0,k]*ss.norm.pdf(self.data,loc=mu_array[k], scale=np.sqrt(sig_array[k]))[:,0]
                for i in range(self.n):
                    S_array[i]=np.random.choice(self.k, 1, p=weights_dist_S[i,:]/sum(weights_dist_S[i,:]))
    #            for i in range(self.n):
    #                #create the vector of weights for the discrete distribution of S
    #                normalising_const=0
    #                weights_dist_S=np.zeros(self.k)
    #                for k in range(self.k):
    #                    weights_dist_S[k]=eta[t][0,k]*ss.norm.pdf(self.data[i],loc=mu_array[k], scale=np.sqrt(sig_array[k]))
    #                    normalising_const+=weights_dist_S[k]
    #
    #                #sample S_i with discrete distribution
    #                S_array[i]=np.random.choice(self.k, 1, p=weights_dist_S/normalising_const)
    ##                S_dist=ss.rv_discrete(values=(np.arange(self.k),weights_dist_S/normalising_const))
    ##                S_array[i]=S_dist.rvs(size=1)
                    N_array[np.int(S_array[i])]+=1
                S.append(S_array)
                N.append(N_array)
    
            del S[0:(burnIn+1)]
            del N[0:(burnIn+1)]
            del eta[0:burnIn]
            del mu[0:burnIn]
            del sigma2[0:burnIn]
            return(eta,mu,sigma2,S,N)
        if self.priorDistLocScale=='IndependencePrior':
            S=[] #vector of allocation of the observation to a cluster
            N=[] #vector of counts of the number of observations in each cluster
            sigma2=[] #scale mixture params
            sigma2.append(np.ones(self.k)) #initialisation of gibbs chain            
            mu=[] #loc mixture params
            mu.append(np.zeros(self.k)) #initialisation of gibbs chain
            eta=[] #weights
            S.append(np.zeros(self.n)) #assign a cluster to each observation S[0] has shape (n,)
            N.append(np.zeros(self.k))
            N[-1][0]=self.k
            for i in range(self.k):
                N[0][i]=np.sum(S[0]==i) #N[0] counts the number of obs in each k clusters
    
            for t in range(numIter): #Warning : N and S are one index late compared to eta and Theta
                if verbose==True:
                    print(t/numIter*100,end="\r")
    
    
                #sample the weights
                eta.append(ss.dirichlet.rvs(alpha=self.priorParamWeight+N[t],size=1))
                #sample the scale and loc parameters for each k group
                c=np.zeros(self.k)
                C=np.zeros(self.k)
                b=np.zeros(self.k)
                B=np.zeros(self.k)
                sig_array=np.zeros(self.k)
                mu_array=np.zeros(self.k)
    
                for i in range(self.k):
    
                    if N[t][i]==0: #empty cluster
                        group_mean=0
                        group_var=0
                    else:
                        data_i=self.data[S[t]==i]
                        group_mean=np.mean(data_i)
                        group_var=np.var(data_i)
    
                    #shape of invgamma
                    c[i]=self.priorParamLocScale[2]+0.5*N[t][i] #F-S eq (6.15)
    
                    #scale of invgamma
                    C[i]=self.priorParamLocScale[3]+0.5*np.sum((data_i-mu[-1][i])**2)
    
                    #sample sigma2
                    sig_array[i]=ss.invgamma.rvs(a=c[i],scale=C[i],size=1)
    
                    #shape of normal dist
                    B[i]=1/(1/self.priorParamLocScale[1]+(1/sig_array[i])*N[-1][i])
    
                    #loc of normal dist
                    b[i]=B[i]*(1/sig_array[i]*N[-1][i]*group_mean+1/self.priorParamLocScale[1]*self.priorParamLocScale[0])
    
                    #sample mu
                    mu_array[i]=ss.norm.rvs(loc=b[i],scale=np.sqrt(B[i]),size=1)
    
    
                sigma2.append(sig_array.copy())
                mu.append(mu_array.copy())
    
                #Update allocations and counts for each observation
                S_array=np.zeros(self.n,dtype=int)
                N_array=np.zeros(self.k,dtype=int)
    
                weights_dist_S=np.zeros((self.n,self.k))
                for k in range(self.k):
                    weights_dist_S[:,k]=np.log(eta[t][0,k])+ss.norm.logpdf(self.data,loc=mu_array[k], scale=np.sqrt(sig_array[k]))[:,0]
                for i in range(self.n):
                    S_array[i]=catDistLogProb(weights_dist_S[i,:])
                    
                    ()
                    #S_array[i]=np.random.choice(self.k, 1, p=weights_dist_S[i,:]/sum(weights_dist_S[i,:]))
    #            for i in range(self.n):
    #                #create the vector of weights for the discrete distribution of S
    #                normalising_const=0
    #                weights_dist_S=np.zeros(self.k)
    #                for k in range(self.k):
    #                    weights_dist_S[k]=eta[t][0,k]*ss.norm.pdf(self.data[i],loc=mu_array[k], scale=np.sqrt(sig_array[k]))
    #                    normalising_const+=weights_dist_S[k]
    #
    #                #sample S_i with discrete distribution
    #                S_array[i]=np.random.choice(self.k, 1, p=weights_dist_S/normalising_const)
    ##                S_dist=ss.rv_discrete(values=(np.arange(self.k),weights_dist_S/normalising_const))
    ##                S_array[i]=S_dist.rvs(size=1)
                    N_array[S_array[i]]+=1
                S.append(S_array.copy())
                N.append(N_array.copy())
    
            del S[0:(burnIn+1)]
            del N[0:(burnIn+1)]
            del eta[0:burnIn]
            del mu[0:burnIn]
            del sigma2[0:burnIn]
            return(eta,mu,sigma2,S,N)
        

        

    def chibEstimator(self, numIterGibbs, burnIn, verbose=False, permutationChib=True):

        eta,mu,sigma2,S,N=self.GibbsSampler(numIterGibbs,burnIn,verbose)

        # Find the MAP
        # augmentedLogPosterior=np.zeros(numIterGibbs-burnIn)
        # for t in range(numIterGibbs-burnIn):
        #     augmentedLogPosterior[t]=self.augmentedLogPosterior(eta[t],np.column_stack((mu[t],sigma2[t])),S[t],N[t])
        # MAP_index=np.argmax(augmentedLogPosterior)

        LogPosterior=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            LogPosterior[t]=self.logPriorWeight(eta[t])+self.logPriorLocScale(np.column_stack((mu[t],sigma2[t])))
        MAP_index=np.argmax(LogPosterior)

        #shape of invgamma
        chib=0
        if permutationChib==True:
            for t in range(numIterGibbs-burnIn):
                for s in set(permutations(np.arange(self.k))):
                    chib+=(1/math.factorial(self.k))*(1/(numIterGibbs-burnIn))*np.exp(self.augmentedLogPosterior(np.array([eta[MAP_index][0,np.array(s)]]),np.column_stack((mu[MAP_index][np.array(s)],sigma2[MAP_index][np.array(s)])), S[t].astype(int), N[t]))
        else:
            for t in range(numIterGibbs-burnIn):
               # print(self.augmentedLogPosterior(np.array([eta[MAP_index]]),np.column_stack((mu[MAP_index],sigma2[MAP_index])), S[t].astype(int), N[t]))
                chib+=(1/(numIterGibbs-burnIn))*np.exp(self.augmentedLogPosterior(np.array([eta[MAP_index][0,:]]),np.column_stack((mu[MAP_index],sigma2[MAP_index])), S[t].astype(int), N[t]))
            
        out=self.logLikelihood(eta[MAP_index],np.column_stack((mu[MAP_index],sigma2[MAP_index])))+self.logPriorWeight(eta[MAP_index])+self.logPriorLocScale(np.column_stack((mu[MAP_index],sigma2[MAP_index])))-np.log(chib)

        return out

    def chibEstimatorV2(self, numIterGibbs, burnIn, verbose=False, permutationChib=True, numRandPermut=0):

        eta,mu,sigma2,S,N=self.GibbsSamplerV2(numIterGibbs,burnIn)

        # Find the MAP
        # augmentedLogPosterior=np.zeros(numIterGibbs-burnIn)
        # for t in range(numIterGibbs-burnIn):
        #     augmentedLogPosterior[t]=self.augmentedLogPosterior(eta[t],np.column_stack((mu[t],sigma2[t])),S[t],N[t])
        # MAP_index=np.argmax(augmentedLogPosterior)

        LogPosterior=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            LogPosterior[t]=self.augmentedLogPosteriorV2(eta[t],mu[t],sigma2[t],S[t],N[t])
        MAP_index=np.argmax(LogPosterior)

        #shape of invgamma
        chib=[]
        if permutationChib==True and numRandPermut==0:
            perm=set(permutations(np.arange(self.k)))
            for t in range(numIterGibbs-burnIn):
                # if t%100==0:
                #     print(t)
                for s in perm:
                    chib.append(-np.log(math.factorial(self.k)*(numIterGibbs-burnIn))+self.augmentedLogPosteriorV2(eta[MAP_index][np.array(s)],mu[MAP_index][np.array(s)],sigma2[MAP_index][np.array(s)], S[t].astype(int), N[t]))
        
        elif permutationChib==True and numRandPermut!=0:
            #add orginial partition to map
            chib.append(-np.log((numRandPermut+1)*(numIterGibbs-burnIn))+self.augmentedLogPosteriorV2(eta[MAP_index],mu[MAP_index],sigma2[MAP_index], S[MAP_index].astype(int), N[MAP_index]))
            permut=list(permutations(np.arange(self.k)))

            for t in range(numIterGibbs-burnIn):
                permutIndex=np.random.choice(math.factorial(self.k),numRandPermut,replace=True)
                
                for s in range(numRandPermut):
                    chib.append(-np.log((numRandPermut+1)*(numIterGibbs-burnIn))+self.augmentedLogPosteriorV2(eta[MAP_index][np.array(permut[permutIndex[s]])],mu[MAP_index][np.array(permut[permutIndex[s]])],sigma2[MAP_index][np.array(permut[permutIndex[s]])], S[t].astype(int), N[t]))
        else:
            for t in range(numIterGibbs-burnIn):
               # print(self.augmentedLogPosterior(np.array([eta[MAP_index]]),np.column_stack((mu[MAP_index],sigma2[MAP_index])), S[t].astype(int), N[t]))
                chib.append(-np.log((numIterGibbs-burnIn))+self.augmentedLogPosteriorV2(eta[MAP_index],mu[MAP_index],sigma2[MAP_index], S[t].astype(int), N[t]))
            
        out=self.logLikelihoodV2(eta[MAP_index],mu[MAP_index],sigma2[MAP_index])+self.logPriorWeightV2(eta[MAP_index])+self.logPriorLocScale(np.column_stack((mu[MAP_index],sigma2[MAP_index])))-logsumexp(chib)

        return out
    
    def augmentedPosteriorSampling(self,S,N):
        eta=ss.dirichlet.rvs(alpha=self.priorParamWeight+N)[0]
        
        c=np.zeros(self.k)
        C=np.zeros(self.k)
        b=np.zeros(self.k)
        B=np.zeros(self.k)
        
        sigma2=np.zeros(self.k)
        mu=np.zeros(self.k)
        for i in range(self.k):
            if N[i]==0: #empty cluster
                group_mean=0
                group_var=0
            else:
                group_mean=np.mean(self.data[S==i])
                group_var=np.var(self.data[S==i])

            #shape of invgamma
            c[i]=self.priorParamLocScale[2]+0.5*N[i] #F-S eq (6.15)

            #scale of invgamma
            C[i]=self.priorParamLocScale[3]+0.5*(N[i]*group_var+((N[i]*self.priorParamLocScale[1])/(N[i]+self.priorParamLocScale[1]))*(group_mean-self.priorParamLocScale[0])**2)
            
            sigma2[i]=ss.invgamma.rvs(a=c[i],scale=C[i])
            #shape of normal dist
            B[i]=sigma2[i]/(N[i]+self.priorParamLocScale[1])

            #loc of normal dist
            b[i]=self.priorParamLocScale[1]*self.priorParamLocScale[0]/(N[i]+self.priorParamLocScale[1])+N[i]*group_mean/(N[i]+self.priorParamLocScale[1])

            #add to post
            mu[i]=ss.norm.rvs(loc=b[i],scale=np.sqrt(B[i]))
        return eta,mu,sigma2
    
    def SIS(self,numSim):
        #Likelihood estimation
        data=self.data
        w=np.zeros(numSim)
        for t in range(numSim):
            np.random.shuffle(data)
            u=np.zeros(self.n)
            S=np.zeros(self.n,dtype=int)
            N=np.zeros(self.k,dtype=int)
            mixtemp=mixnorm1d(data=data[0],k=self.k,priorDistWeight=self.priorDistWeight,priorParamWeight=self.priorParamWeight,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale)
            N[0]+=1
            #print(data)
            Ntemp=np.zeros(self.k)
            Ntemp[0]=mixtemp.n
            u[0]=np.log(self.k)+np.log(self.priorParamWeight[0]/(self.k*self.priorParamWeight[0]))+mixtemp.logLikelihoodAllocations(np.array([0]),Ntemp)
            for i in range(1,self.n):
                logProbs=np.zeros(self.k)
                for k in range(self.k):
                    logProbs[k]+=np.log((N[k]+self.priorParamWeight[0])/(i+self.k*self.priorParamWeight[k]))
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=k#pretend y is in cluster k
                    ytemp=data[:(i+1)][Stemp==k]
                    mixtemp=mixnorm1d(data=ytemp,k=self.k,priorDistWeight=self.priorDistWeight,priorParamWeight=self.priorParamWeight,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale)
                    Stemp=np.zeros(mixtemp.n,dtype=int).copy()
                    Ntemp=np.zeros(self.k)
                    Ntemp[0]=mixtemp.n
                    logProbs[k]=logProbs[k]+mixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=self.n+1
                    ytemp=data[:(i+1)][Stemp==k]
                    mixtemp=mixnorm1d(data=ytemp,k=self.k,priorDistWeight=self.priorDistWeight,priorParamWeight=self.priorParamWeight,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale)
                    Stemp=np.zeros(mixtemp.n,dtype=int)
                    Ntemp=np.zeros(self.k)
                    Ntemp[0]=mixtemp.n
                    logProbs[k]=logProbs[k]-mixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    

                sampledCluster=catDistLogProb(logProbs)
                S[i]=sampledCluster
                N[sampledCluster]+=1
                
                
                u[i]=logsumexp(logProbs)
            w[t]=np.sum(u)
        llk=-np.log(numSim)+logsumexp(w)
        #print('llk uncertainty sd', 1/np.sqrt(numSim)*np.exp(0.5*log_domain_var(w)-llk))
        logvar=-np.log(numSim)+logsumexp(2*w) + np.log(1-np.exp((-np.log(numSim**2)+2*logsumexp(w))-(-np.log(numSim)+logsumexp(2*w))))
        #print('var', logvar)
        return llk,1/np.sqrt(numSim)*np.exp(0.5*logvar-llk)
    
    def SISv2(self,numSim): #clone of SIS, without returning variance estimator
        #Likelihood estimation
        data=self.data
        w=np.zeros(numSim)
        for t in range(numSim):
            np.random.shuffle(data)
            u=np.zeros(self.n)
            S=np.zeros(self.n,dtype=int)
            N=np.zeros(self.k,dtype=int)
            mixtemp=mixnorm1d(data=data[0],k=self.k,priorDistWeight=self.priorDistWeight,priorParamWeight=self.priorParamWeight,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale)
            N[0]+=1
            #print(data)
            Ntemp=np.zeros(self.k)
            Ntemp[0]=mixtemp.n
            u[0]=np.log(self.k)+np.log(self.priorParamWeight[0]/(self.k*self.priorParamWeight[0]))+mixtemp.logLikelihoodAllocations(np.array([0]),Ntemp)
            for i in range(1,self.n):
                logProbs=np.zeros(self.k)
                for k in range(self.k):
                    logProbs[k]+=np.log((N[k]+self.priorParamWeight[0])/(i+self.k*self.priorParamWeight[k]))
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=k#pretend y is in cluster k
                    ytemp=data[:(i+1)][Stemp==k]
                    mixtemp=mixnorm1d(data=ytemp,k=self.k,priorDistWeight=self.priorDistWeight,priorParamWeight=self.priorParamWeight,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale)
                    Stemp=np.zeros(mixtemp.n,dtype=int).copy()
                    Ntemp=np.zeros(self.k)
                    Ntemp[0]=mixtemp.n
                    logProbs[k]=logProbs[k]+mixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    
                    Stemp=S[:(i+1)].copy()
                    Stemp[i]=self.n+1
                    ytemp=data[:(i+1)][Stemp==k]
                    mixtemp=mixnorm1d(data=ytemp,k=self.k,priorDistWeight=self.priorDistWeight,priorParamWeight=self.priorParamWeight,priorDistLocScale=self.priorDistLocScale,priorParamLocScale=self.priorParamLocScale)
                    Stemp=np.zeros(mixtemp.n,dtype=int)
                    Ntemp=np.zeros(self.k)
                    Ntemp[0]=mixtemp.n
                    logProbs[k]=logProbs[k]-mixtemp.logLikelihoodAllocations(Stemp,Ntemp)
                    

                sampledCluster=catDistLogProb(logProbs)
                S[i]=sampledCluster
                N[sampledCluster]+=1
                
                
                u[i]=logsumexp(logProbs)
            w[t]=np.sum(u)
        llk=-np.log(numSim)+logsumexp(w)
        #print('llk uncertainty sd', 1/np.sqrt(numSim)*np.exp(0.5*log_domain_var(w)-llk))
        logvar=-np.log(numSim)+logsumexp(2*w) + np.log(1-np.exp((-np.log(numSim**2)+2*logsumexp(w))-(-np.log(numSim)+logsumexp(2*w))))
        #print('var', logvar)
        return llk
    
    def BridgeSampling(self,numIterGibbs, burnIn, M0): #Algorithm 3 from https://projecteuclid.org/journals/brazilian-journal-of-probability-and-statistics/volume-33/issue-4/Keeping-the-balanceBridge-sampling-for-marginal-likelihood-estimation-in-finite/10.1214/19-BJPS446.full
        eta_post,mu_post,sigma2_post,S_post,N_post=self.GibbsSamplerV2(numIterGibbs,burnIn)
        M=numIterGibbs-burnIn
        L=numIterGibbs-burnIn
        if M0>M:
            sys.exit('M0 must be lower than M')
        
        qComponents=np.random.choice(M,size=M0)
        
        permut=list(permutations(np.arange(self.k)))
        #Sample from q
        eta_q=[]
        mu_q=[]
        sigma2_q=[]
        for t in range(L):
            #Sample a component
            compoIndex=np.random.choice(M0)
            #Sample a permutation
            permutIndex=np.random.choice(math.factorial(self.k))
            p=np.array(permut[permutIndex])
            #Permute S:
            S_q=S_post[qComponents[compoIndex]].copy()
            w=S_post[qComponents[compoIndex]].copy()
            N_q=N_post[qComponents[compoIndex]].copy()
            for i in range(self.k):
                N_q[p[i]]=N_post[qComponents[compoIndex]][i]
                #print(p)
                S_q[w==i]=p[i]
            
            #N_q=N_q[p]
            out=self.augmentedPosteriorSampling(S_q,N_q)
            eta_q.append(out[0].copy())
            mu_q.append(out[1].copy())
            sigma2_q.append(out[2].copy())
            #print(mu_q[-1],S_q,N_q)
        logProb_q_qsamp=np.zeros(L)
        logProb_q_qsamp_compo=[]
        logProb_q_postsamp=np.zeros(M)
        logProb_q_postsamp_compo=[]

        logProb_post_qsamp=np.zeros(L)
        logProb_post_postsamp=np.zeros(M)
        if M==L:
             for t in range(M):
                 #print(t)
                 for i in range(M0):
                     for s in permut:
                         #print(t)
                         #print(eta_q[t])
                         logProb_q_qsamp_compo.append(-np.log(math.factorial(self.k)*(M0))+self.augmentedLogPosteriorV2(eta_q[t][np.array(s)],mu_q[t][np.array(s)],sigma2_q[t][np.array(s)], S_post[qComponents[i]].astype(int), N_post[qComponents[i]]))
                         logProb_q_postsamp_compo.append(-np.log(math.factorial(self.k)*(M0))+self.augmentedLogPosteriorV2(eta_post[t][np.array(s)],mu_post[t][np.array(s)],sigma2_post[t][np.array(s)], S_post[qComponents[i]].astype(int), N_post[qComponents[i]]))
                 logProb_q_qsamp[t]=logsumexp(logProb_q_qsamp_compo)       
                 logProb_q_postsamp[t]=logsumexp(logProb_q_postsamp_compo)
                 logProb_q_qsamp_compo.clear()
                 logProb_q_postsamp_compo.clear()
                 logProb_post_qsamp[t]=self.logLikelihoodV2(eta_q[t],mu_q[t],sigma2_q[t])+self.logPriorWeightV2(eta_q[t])+self.logPriorLocScale(np.column_stack((mu_q[t],sigma2_q[t])))
                 logProb_post_postsamp[t]=self.logLikelihoodV2(eta_post[t],mu_post[t],sigma2_post[t])+self.logPriorWeightV2(eta_post[t])+self.logPriorLocScale(np.column_stack((mu_post[t],sigma2_post[t])))
             
        #First iteration : Simple IS
        BS=[]
        BS.append(0)
        BS.append(-np.log(L)+logsumexp(logProb_post_qsamp-logProb_q_qsamp))
        #print(BS)
        
        while np.abs(BS[-1]-BS[-2])>0.01:
            numerator=np.zeros(L)
            denominator=np.zeros(M)
            if M==L:
                ESS=M/2
                for t in range(M):
                    numerator[t]=logProb_post_qsamp[t]-logsumexp(np.array([np.log(L)+logProb_q_qsamp[t],np.log(ESS)+logProb_post_qsamp[t]-BS[-1]]))
                    denominator[t]=logProb_q_postsamp[t]-logsumexp(np.array([np.log(L)+logProb_q_postsamp[t],np.log(ESS)+logProb_post_postsamp[t]-BS[-1]]))
            BS.append(-np.log(L)+logsumexp(numerator)+np.log(M)-logsumexp(denominator))
            #print(BS)
                                                            
                                                                
                                                                      
        #print(logProb_q_postsamp,logProb_q_qsamp)
        return BS[-1]
    
    
    
    def chibEstimatorPartitions(self,numIterGibbs,burnIn):
        eta,mu,sigma2,S,N=self.GibbsSamplerV2(numIterGibbs,burnIn)
        
        #Find the MAP partition
        UnormalisedLogPosterior=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            UnormalisedLogPosterior[t]=self.logPosteriorPartition(S[t],N[t])
        MAP_index=np.argmax(UnormalisedLogPosterior)
        
        
        #MAP_index=np.random.choice(numIterGibbs-burnIn)
        #Compute the MC estimator of the log posterior
        #print(S[MAP_index],N[MAP_index])
        PosteriorCount=0
        MAPCanonicalPartition=canonical_form_partition(S[MAP_index])
        #print(S[MAP_index],MAPCanonicalPartition)
        h=np.zeros(numIterGibbs-burnIn) #following chib 1995 p.1316 notation
        for t in range(numIterGibbs-burnIn):
            #print(S[t]
            if (canonical_form_partition(S[t])==MAPCanonicalPartition).all():
                #print('True')
                #print(S[t])
                h[t]=1
                PosteriorCount+=1
        logPosterior=-np.log(numIterGibbs-burnIn)+np.log(PosteriorCount)
        ##following chib 1995 p.1316 :
        variances=(h-(PosteriorCount)/(numIterGibbs-burnIn))**2
        q=10
        var=0
        G=numIterGibbs-burnIn
        var=(1/G)*(np.sum(variances)/G)
        for s in range(q):
            var+=(1/G)*(1-(s+1)/(q+1)*(1/G)*np.sum(variances[(s+1):]))
        varLog=((PosteriorCount)/(numIterGibbs-burnIn))**(-2)*var
        #print(logPosterior)
        #out=self.logLikelihoodAllocations(S[MAP_index],N[MAP_index])+np.log(math.factorial(self.k))+self.logPriorAllocation(S[MAP_index],N[MAP_index])-logPosterior
        out=self.logLikelihoodAllocations(S[MAP_index],N[MAP_index])+self.logEPPF(N[MAP_index])-logPosterior
        #print(out,varLog)
        return out
        
    def chibEstimatorPartitionsNotMAP(self,numIterGibbs,burnIn):
        eta,mu,sigma2,S,N=self.GibbsSamplerV2(numIterGibbs,burnIn)
        
        #Find the MAP partition
        listPartitions=[]
        for t in range(numIterGibbs-burnIn):
            listPartitions.append(canonical_form_partition(S[t]))
        mtx = np.matrix(listPartitions)
        values, idx, counts = np.unique(mtx, return_index=True, return_counts=True, axis=0)
        postMostFreqPart=np.max(counts)
        mstFrqPart=values[counts==postMostFreqPart,][0]
        mstFrqIdx=int(idx[counts==postMostFreqPart][0])
        print(postMostFreqPart,mstFrqPart,mstFrqIdx)
        logPosterior=-np.log(numIterGibbs-burnIn)+np.log(postMostFreqPart)
        #print(logPosterior)
        #out=self.logLikelihoodAllocations(S[MAP_index],N[MAP_index])+np.log(math.factorial(self.k))+self.logPriorAllocation(S[MAP_index],N[MAP_index])-logPosterior
        out=self.logLikelihoodAllocations(S[mstFrqIdx],N[mstFrqIdx])+self.logEPPF(N[mstFrqIdx])-logPosterior
        print(out)
        return out

    # def chibEstimatorPartitionsSet(self,numIterGibbs,burnIn,NumPart):
    #     eta,mu,sigma2,S,N=self.GibbsSamplerV2(numIterGibbs,burnIn)
        
    #     #Find the MAP partition
    #     UnormalisedLogPosterior=np.zeros(numIterGibbs-burnIn)
    #     for t in range(numIterGibbs-burnIn):
    #         UnormalisedLogPosterior[t]=self.logPosteriorPartition(S[t],N[t])
    #     #Find the indices of the top NumPart Partitions
    #     #print(UnormalisedLogPosterior)
    #     UniqueUnormalisedLogPosterior,indicesList=np.unique(UnormalisedLogPosterior, return_index=True) #breaks ties and sorts and gives the indices of the original output
    #     MAP_index=indicesList[-NumPart:]
    #     #MAP_index=np.argpartition(UnormalisedLogPosterior, -NumPart)[-NumPart:]#Problem : will count several times the same partitions
    #     #print(UniqueUnormalisedLogPosterior)
    #     #print(MAP_index)
    #     #print(indicesList)
    #     #MAP_index=np.unique(MAP_index)
    #     #print(MAP_index)
        
    #     PosteriorCount=0
    #     MAPCanonicalPartition=[]
    #     for i in range(len(MAP_index)):
    #         #print(MAP_index[i])
    #         MAPCanonicalPartition.append(canonical_form_partition(S[MAP_index[i]]))
            
    #     #print(MAPCanonicalPartition)        
    #     for t in range(numIterGibbs-burnIn):
    #         #print(S[t]
    #         count=0
    #         for l in MAPCanonicalPartition:
                
    #             if (canonical_form_partition(S[t])==l).all():
    #                 #print(l,S[t])
    #                 #print('True',t)
    #                 count+=1
    #                 PosteriorCount+=1
    #                 #break;
    #             if count>1:
    #                 print('ee')
    #     logPosterior=-np.log(numIterGibbs-burnIn)+np.log(PosteriorCount)
    #     #print(logPosterior)
    #     #out=self.logLikelihoodAllocations(S[MAP_index],N[MAP_index])+np.log(math.factorial(self.k))+self.logPriorAllocation(S[MAP_index],N[MAP_index])-logPosterior
    #     logLikelihood=np.zeros(len(MAP_index))
    #     logPrior=np.zeros(len(MAP_index))
    #     for i in range(len(MAP_index)):
    #         logLikelihood[i]=self.logLikelihoodAllocations(S[MAP_index[i]],N[MAP_index[i]])
    #         logPrior[i]=self.logEPPF(N[MAP_index[i]])
    #     logLikelihood=logsumexp(logLikelihood)
    #     logPrior=logsumexp(logPrior)
    #     out=logLikelihood+logPrior-logPosterior
    #     print(logPrior,logLikelihood,logPosterior)
    #     print(out)
    #     return MAP_index,UnormalisedLogPosterior,MAPCanonicalPartition
        
    def chibEstimatorAllocations(self,numIterGibbs,burnIn, verbose=False):
        eta,mu,sigma2,S,N=self.GibbsSampler(numIterGibbs,burnIn,verbose)
        
        #Find the MAP
        logPosterior=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            logPosterior[t]=self.logPriorAllocation(S[t], N[t])+self.logLikelihoodAllocations(S[t],N[t])
        MAP=np.argmax(logPosterior)
        
        logPosterior=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            
            for i in range(self.n):
                logProbs=np.zeros(self.k)
                for k in range(self.k):
                    logProbs[k]=np.log(eta[t][0,k])+ss.norm.logpdf(self.data[i],loc=mu[t][k], scale=np.sqrt(sigma2[t][k]))
                logPosterior[t]+=logProbs[np.int(S[MAP][i])]-logsumexp(logProbs)
                
        logPosterior=-np.log(numIterGibbs-burnIn)+logsumexp(logPosterior)
        
        return self.logLikelihoodAllocations(S[MAP],N[MAP])+self.logPriorAllocation(S[MAP],N[MAP])-logPosterior
    def nestedSampling(self, numIter, activeSetSize, GibbsStep,verbose=False):

        eta=[]
        mu=[]
        sigma2=[]

        #Comupting the log likelihood of the active setA
        L=np.zeros(activeSetSize)
        for i in range(activeSetSize):
            eta.append(self.priorSimWeight())
            sigma2.append(self.priorSimScale())
            mu.append(self.priorSimLoc(sigma2[i]))
            L[i]=self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i])))
#        eta=np.reshape(np.zeros(activeSetSize*self.k),(activeSetSize,self.k))
#        mu=np.reshape(np.zeros(activeSetSize*self.k),(activeSetSize,self.k))
#        sigma2=np.reshape(np.zeros(activeSetSize*self.k),(activeSetSize,self.k))
#
#        eta=self.priorSimWeight(activeSetSize)
#        sigma2=self.priorSimScale(activeSetSize)
#        print(eta.shape[1])
#        print(sigma2.shape)
#        print(self.logLikelihood(eta,np.column_stack((mu,sigma2))))
#        for i in range(activeSetSize):
#            mu[i,:]=self.priorSimLoc(sigma2[i,:])

        Z=0 #evidence
        X=np.zeros(numIter) #prior volume
        llk=np.zeros(numIter) #log-likelihood chain

        for t in range(numIter):

            #Computing the prior volume
            X[t]=np.exp(-(t)/activeSetSize)


            minIndex=np.argmin(L)
            llk[t]=L[minIndex]

            #sample randomly one of the survivors for the init state of gibbs
            while True:
                GibbsStart=int(np.random.choice(activeSetSize,1))
                if GibbsStart!=minIndex:
                    break
            etaGibbs=[]
            etaGibbs.append(eta[GibbsStart])
            muGibbs=[]
            muGibbs.append(mu[GibbsStart])
            sigma2Gibbs=[]
            sigma2Gibbs.append(sigma2[GibbsStart])

            for i in range(GibbsStep):
                #Sample eta with MH within Gibbs where the proposal distribution is the prior
                etaProposal=ss.dirichlet.rvs(self.priorParamWeight)
                if self.logLikelihood(etaProposal,np.column_stack((muGibbs[i],sigma2Gibbs[i])))>L[minIndex]:
                    etaGibbs.append(etaProposal)
                else:
                    etaGibbs.append(etaGibbs[i])

                #Update mu with MH within gibbs where the proposal is the prior
                muProposal=self.priorSimLoc(sigma2Gibbs[i])
                if self.logLikelihood(etaGibbs[i+1],np.column_stack((muProposal,sigma2Gibbs[i])))>L[minIndex]:
                    muGibbs.append(muProposal)
                else:
                    muGibbs.append(muGibbs[i])

                #Update sigma2 with MH within gibbs taking the prior as proposal
                sigma2Proposal=self.priorSimScale()
                #print(muGibbs[t+1],sigma2Proposal)

                if self.logLikelihood(etaGibbs[i+1],np.column_stack((muGibbs[i+1],sigma2Proposal)))>L[minIndex]:
                    if np.log(np.random.uniform(size=1))<self.logPriorLoc(muGibbs[i+1],sigma2Proposal)+self.logPriorScale(sigma2Proposal):
                        sigma2Gibbs.append(sigma2Proposal)
                    else:
                        sigma2Gibbs.append(sigma2Gibbs[i])
                else:
                    sigma2Gibbs.append(sigma2Gibbs[i])
            eta[minIndex]=etaGibbs[-1]
            mu[minIndex]=muGibbs[-1]
            sigma2[minIndex]=sigma2Gibbs[-1]
            L[minIndex]=self.logLikelihood(eta[minIndex],np.column_stack((mu[minIndex],sigma2[minIndex])))

            if(t>=2):
                mm=np.max(np.log(-np.diff(X[np.arange(t)]))+llk[np.arange(t-1)+1])
                Z=mm+np.log(np.sum(np.exp(np.log(-np.diff(X[np.arange(t)]))+llk[np.arange(t-1)+1]-mm)))
                stopping=np.max(L)+np.log(X[t])
                #print(llk)
                if verbose==True:
                    print(Z,stopping,end="\r")
                if stopping-Z<0.000001:
                    if verbose==True:
                        print("Converged : ", Z, stopping-Z)
                    break
        if verbose==True:
            print("nombre max d'itérations atteint : ", t)
        return Z

    def SMC(self,numParticles, temperatures): #####ATTENTION LETAPE RESAMPLING EST SUPPRIMEE
        eta=[]
        mu=[]
        sigma2=[]
        W=np.ones(numParticles)/numParticles
        Z=0

        #First step : sample from prior
        for i in range(numParticles):
            eta.append(self.priorSimWeight())
            sigma2.append(self.priorSimScale())
            mu.append(self.priorSimLoc(sigma2[i]))
        for t in range(len(temperatures)-1):

            W=np.log(W)
            #Reweight
            for i in range(numParticles):
                W[i]=W[i]+(temperatures[t+1]-temperatures[t])*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i])))
            #Update log marginal
            Z+=logsumexp(W)


            ESS=1/np.sum((np.exp(W)/np.exp(logsumexp(W)))**2)
            print(ESS)
            #Resample AND Move (MH within) Gibbs step
            for i in range(numParticles):
                if ESS<0.8*numParticles :
                    sampIndex=catDistLogProb(W)
                    eta[i]=eta[sampIndex]
                    mu[i]=mu[sampIndex]
                    sigma2[i]=sigma2[sampIndex]

                for j in range(1):
                    #Sample eta proposal from prior
                    etaProp=ss.dirichlet.rvs(self.priorParamWeight)
                    if np.log(np.random.uniform(size=1))<temperatures[t+1]*self.logLikelihood(etaProp,np.column_stack((mu[i],sigma2[i])))-temperatures[t+1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i]))):
                        eta[i]=etaProp

                    #Sample mu proposal from prior
                    muProp=self.priorSimLoc(sigma2[i])
                    if np.log(np.random.uniform(size=1))<temperatures[t+1]*self.logLikelihood(eta[i],np.column_stack((muProp,sigma2[i])))-temperatures[t+1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i]))):
                        mu[i]=muProp

                    #Sample sigma2 proposal from prior
                    sigma2Prop=self.priorSimScale()
                    if np.log(np.random.uniform(size=1))<temperatures[t+1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2Prop)))+self.logPriorLoc(mu[i],sigma2Prop)-temperatures[t+1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i])))-self.logPriorLoc(mu[i],sigma2[i]):
                        sigma2[i]=sigma2Prop
            W=np.exp(W)/np.exp(logsumexp(W))
            #print(np.exp(-logsumexp(2*W)))

        return(Z)

        
    def adaptiveSMC(self,numParticles,numGibbsStep,ESSThreshold=0.8,maxIterBissection=10000):
        TOL=1
        temp=[]
        temp.append(0)
        eta=[]
        mu=[]
        sigma2=[]
        
        Z=0
        #First step : sample from prior
        for i in range(numParticles):
            eta.append(self.priorSimWeight())
            sigma2.append(self.priorSimScale())
            mu.append(self.priorSimLoc(sigma2[i]))
            

        while temp[-1]<1:
           
            if temp[-1]!=0:
                #Mutation step
                #print('Mutation')
                moved_eta =0
                moved_mu =0
                moved_sig =0
                for i in range(numParticles):
                    
                    for j in range(numGibbsStep):
                        #Sample eta proposal from prior
                        # etaProp=ss.dirichlet.rvs(self.priorParamWeight)
                        # if np.log(np.random.uniform(size=1))<temp[-1]*self.logLikelihood(etaProp,np.column_stack((mu[i],sigma2[i])))-temp[-1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i]))):
                        #     eta[i]=etaProp
                        #     moved_eta+=1
                        
                        etaProp=ss.multivariate_normal.rvs(mean=eta[i][0][:(self.k-1)],cov=0.01*np.identity(self.k-1))
                        etaProp=np.append(etaProp,1-np.sum(etaProp))
                        if (etaProp<0).any() or (etaProp>1).any() or np.sum(etaProp)>1:
                            eta[i]=eta[i]
                        elif np.log(np.random.uniform(size=1))<temp[-1]*self.logLikelihood(np.array([etaProp]),np.column_stack((mu[i],sigma2[i])))+self.logPriorWeightV2(etaProp)-temp[-1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i])))-self.logPriorWeight(eta[i]):
                            eta[i]=np.array([etaProp])
                            moved_eta+=1
                        # #Sample mu proposal from prior
                        # muProp=self.priorSimLoc(sigma2[i])
                        # if np.log(np.random.uniform(size=1))<temp[-1]*self.logLikelihood(eta[i],np.column_stack((muProp,sigma2[i])))-temp[-1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i]))):
                        #     mu[i]=muProp
                        #     moved_mu+=1   
                            
                        muProp=ss.multivariate_normal.rvs(mean=mu[i],cov=0.1*np.identity(self.k))
                        if np.log(np.random.uniform(size=1))<temp[-1]*self.logLikelihood(eta[i],np.column_stack((muProp,sigma2[i])))+self.logPriorLoc(muProp, sigma2[i])-temp[-1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i])))-self.logPriorLoc(mu[i], sigma2[i]):
                            mu[i]=muProp
                            moved_mu+=1
                            
                        sigma2Prop=ss.multivariate_normal.rvs(mean=sigma2[i],cov=0.1*np.identity(self.k))
                        if (sigma2Prop<=0).any():
                            sigma2[i]=sigma2[i]
                        elif np.log(np.random.uniform(size=1))<temp[-1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2Prop)))+self.logPriorLoc(mu[i],sigma2Prop)+self.logPriorScale(sigma2Prop)-temp[-1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i])))-self.logPriorLoc(mu[i],sigma2[i])-self.logPriorScale(sigma2[i]):
                            sigma2[i]=sigma2Prop
                            moved_sig+=1
                        #Sample sigma2 proposal from prior
                        # sigma2Prop=self.priorSimScale()
                        # if np.log(np.random.uniform(size=1))<temp[-1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2Prop)))+self.logPriorLoc(mu[i],sigma2Prop)-temp[-1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i])))-self.logPriorLoc(mu[i],sigma2[i]):
                        #     sigma2[i]=sigma2Prop
                        #     moved_sig+=1
                #print('moved_sig',moved_sig/(numParticles*numGibbsStep),'moved_mu',moved_mu/(numParticles*numGibbsStep),'moved_eta',moved_eta/(numParticles*numGibbsStep))
            #Find the next temperature adaptively
            llkParticles=np.zeros(numParticles)
            for j in range(numParticles):
                llkParticles[j]=self.logLikelihood(eta[j],np.column_stack((mu[j],sigma2[j])))
            #Try temperature = 1
            wtemp=(1-temp[-1])*llkParticles
            Wtemp=wtemp-logsumexp(wtemp)
            logESS=-logsumexp(2*Wtemp)
            if logESS>np.log(ESSThreshold*numParticles):
                temp.append(1)
                
            #else do bissection algorithm
            else:
                #print('start')
                a=temp[-1]
                b=1
                l=0
                while l<maxIterBissection:
                    tempcand=(a+b)/2
                    #print(tempcand)
                    
                    wtemp=(tempcand-temp[-1])*llkParticles
                    Wtemp=wtemp-logsumexp(wtemp)
                    logESS=-logsumexp(2*Wtemp)
                    #print(np.exp(logESS))
                    #if np.abs(np.exp(logESS)-ESSThreshold*numParticles)<TOL or ((b-a)/2)<TOL:
                    if np.abs(np.exp(logESS)-ESSThreshold*numParticles)<TOL:
                        #print('true',np.exp(logESS))
                        break
                    else:
                        if logESS>np.log(ESSThreshold*numParticles): #need to increase the temp
                            a=tempcand
                        else:
                            b=tempcand
                    l=l+1
                    
                temp.append(tempcand)
            #print(temp[-1],np.exp(logESS))
            
            #Reweight
            #print((temp[-1],temp[-2]))
            w=(temp[-1]-temp[-2])*llkParticles
            Z+=-np.log(numParticles)+logsumexp(w)
            #print(Z)
            
            #Resample
            for i in range(numParticles):
                sampIndex=catDistLogProb(w)
                eta[i]=eta[sampIndex]
                mu[i]=mu[sampIndex]
                sigma2[i]=sigma2[sampIndex]
        return Z

    def SMC2(self,numParticles, T): #####ATTENTION LETAPE RESAMPLING EST SUPPRIMEE
        eta=[]
        mu=[]
        sigma2=[]
        W=np.ones(numParticles)/numParticles
        Z=0

        temp=[]
        temp.append(0)
        #First step : sample from prior
        for i in range(numParticles):
            eta.append(self.priorSimWeight())
            sigma2.append(self.priorSimScale())
            mu.append(self.priorSimLoc(sigma2[i]))
        t=0
        while True:

            if t/T<0.20:
                temp.append(temp[t]+1/T)
                #print(temp[t])
            elif t/T<0.40 and temp[t-1]<0.4-2/T:
                temp.append(temp[t]+2/T)
                #print(temp[t])
            elif t/T<1 and temp[t-1]<1-3/T:
                temp.append(temp[t]+3/T)
                #print(temp[t])
            else:
                temp.append(1)
                #print(temp[t])

            W=np.log(W)
            #Reweight
            for i in range(numParticles):
                W[i]=W[i]+(temp[t+1]-temp[t])*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i])))
            #Update log marginal
            Z+=logsumexp(W)


            ESS=1/np.sum((np.exp(W)/np.exp(logsumexp(W)))**2)
            #print(ESS)
            #Resample AND Move (MH within) Gibbs step
            for i in range(numParticles):
                if ESS<0.8*numParticles :
                    sampIndex=catDistLogProb(W)
                    eta[i]=eta[sampIndex]
                    mu[i]=mu[sampIndex]
                    sigma2[i]=sigma2[sampIndex]

                for j in range(1):
                    #Sample eta proposal from prior
                    etaProp=ss.dirichlet.rvs(self.priorParamWeight)
                    if np.log(np.random.uniform(size=1))<temp[t+1]*self.logLikelihood(etaProp,np.column_stack((mu[i],sigma2[i])))-temp[t+1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i]))):
                        eta[i]=etaProp

                    #Sample mu proposal from prior
                    muProp=self.priorSimLoc(sigma2[i])
                    if np.log(np.random.uniform(size=1))<temp[t+1]*self.logLikelihood(eta[i],np.column_stack((muProp,sigma2[i])))-temp[t+1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i]))):
                        mu[i]=muProp

                    #Sample sigma2 proposal from prior
                    sigma2Prop=self.priorSimScale()
                    if np.log(np.random.uniform(size=1))<temp[t+1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2Prop)))+self.logPriorLoc(mu[i],sigma2Prop)-temp[t+1]*self.logLikelihood(eta[i],np.column_stack((mu[i],sigma2[i])))-self.logPriorLoc(mu[i],sigma2[i]):
                        sigma2[i]=sigma2Prop
            W=np.exp(W)/np.exp(logsumexp(W))
            t+=1
            if (temp[-1]==1):
                break
                return(Z)

            #print(np.exp(-logsumexp(2*W)))

        return(Z)

    def harmonicMean(self,numIterGibbs,burnIn):
        eta,mu,sigma2,S,N=self.GibbsSamplerCondConjPrior(numIterGibbs,burnIn)
        logInvLlk=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            logInvLlk[t]=-self.logLikelihood(eta[t],np.column_stack((mu[t],sigma2[t])))
        return -(np.log(1/(numIterGibbs-burnIn))+logsumexp(logInvLlk))

    def arithmeticMean(self,numSim):
        etaSim=[]
        muSim=[]
        sigma2Sim=[]
        logLlk=np.zeros(numSim)
        for t in range(numSim):
            etaSim.append(self.priorSimWeight())
            sigma2Sim.append(self.priorSimScale())
            muSim.append(self.priorSimLoc(sigma2Sim[-1]))
            logLlk[t]=self.logLikelihood(etaSim[-1],np.column_stack((muSim[-1],sigma2Sim[-1])))
        return -np.log(numSim)+logsumexp(logLlk)





class DPmixnorm1d:
    def __init__(self, data=None, priorDistLocScale=None,
                 priorParamLocScale=None, priorDistAlpha=None, priorParamAlpha=None):
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

        if priorDistAlpha is None:
            self.priorDistAlpha='Gamma'
        else:
            self.priorDistAlpha=priorDistAlpha

        if priorParamAlpha is None:
            if self.priorDistAlpha=='Gamma':
                self.priorParamAlpha=np.array([1,1]) #Shape and RATE : watch out with def of wiki and scipy
        else:
            self.priorParamAlpha=priorParamAlpha

    def priorSimAlpha(self):
        if self.priorDistAlpha=='Gamma':
            return ss.gamma.rvs(a=self.priorParamAlpha[0],scale=1/self.priorParamAlpha[1]) #Shape and RATE : watch out with def of wiki and scipy
    def mixCompLlk(self,mu,sigma2):
        return ss.norm.logpdf(self.data,loc=mu,scale=np.sqrt(sigma2))
    def logLikelihood(self,mu,sigma2):
        out=0
        for i in range(self.n):
            out+=ss.norm.logpdf(self.data[i],loc=mu[i],scale=np.sqrt(sigma2[i]))
        return out


    def NealsGibbsSamplerConjugate(self,numIter,burnIn):
        if self.priorDistLocScale=='Conjugate':



            C=[]
            N=[]
            mu=[]
            concParam=[] #concentration parameter of the DP
            eta=[]#latent var for sampling concPram of DP
            concParam.append(self.priorSimAlpha())#initialise alpha with a draw from the prior
            mu.append(np.zeros(self.n))
            sigma2=[]
            sigma2.append(np.zeros(self.n))
            C.append(np.zeros(self.n,dtype=int)) #Assign all the observation to cluster 0
            N.append(np.zeros(self.n,dtype=int))#Variable that counts the number of obs per cluster
            N[0][0]=self.n #all the obs are in the first cluster a prioro

            numberActiveClusters=[] #number of active clusters
            for t in range(numIter):
                print(t)
                #print(N[t])
                for i in range(self.n):
                    #print(N[-1])
                    if N[-1][C[-1][i]]==1: #i.e no other obs assigned to this cluster
                        #print('erase cluster ', C[-1][i])
                        mu[-1][C[-1][i]]=0 # remove the parameter of this cluster : kille the cluster
                        sigma2[-1][C[-1][i]]=0

                    #Draw a new value for c_i
                    N[-1][C[-1][i]]+=-1 #Remove the previous allocation of y_i
                    activeClusters=np.unique(C[-1][np.arange(len(C[-1]))!=i])
                    nStar=len(activeClusters)
                    samplingWeights=np.zeros(nStar+1) #log weights to sample c_i
                    for j in range(nStar): #iterate through all the active clusters
                        samplingWeights[j]=np.log(N[-1][activeClusters[j]]/(self.n-1+concParam[-1]))+ss.norm.logpdf(self.data[i],loc=mu[-1][activeClusters[j]],scale=np.sqrt(sigma2[-1][activeClusters[j]]))
                    mu0=self.priorParamLocScale[0]
                    nu=self.priorParamLocScale[1]
                    alpha=self.priorParamLocScale[2]
                    beta=self.priorParamLocScale[3]
                    beta_n=beta+nu/(2*(nu+1))*(self.data[i]-mu0)**2
                    samplingWeights[nStar]=np.log(concParam[-1]/(self.n-1+concParam[-1]))-0.5*np.log(2*np.pi*beta_n)+alpha*np.log(beta/beta_n)+0.5*np.log(nu/(nu+1))+loggamma(alpha+0.5)-loggamma(alpha)

                    sampledIndex=catDistLogProb(samplingWeights)
                    if sampledIndex==nStar: #create a new cluster
                        emptyClusterIndex=next((idx for idx, val in np.ndenumerate(N[-1]) if val==0)) #Find the first zero cluster
                        #print('add cluster ',emptyClusterIndex)
                        N[-1][emptyClusterIndex]+=1
                        C[-1][i]=emptyClusterIndex[0]
                        #print('active cluster ', np.unique(C[-1]))

                        #Sample a new parameter value from the posterior (given y_i only)
                        ##Sample sigma2 from inverse gamma alpha_{n=1}=alpha+1/2 et beta_{n=1}=beta_n
                        sigma2[-1][emptyClusterIndex]=ss.invgamma.rvs(a=alpha+1/2,scale=beta_n)
                        ##Sample mu|sigma2 from normal with appropriate params
                        mu[-1][emptyClusterIndex]=ss.norm.rvs(loc=(self.data[i]+nu*mu0/self.n+nu),scale=np.sqrt(sigma2[-1][emptyClusterIndex]/(nu+1)))
                    else:
                        sampledCluster=activeClusters[sampledIndex]
                        N[-1][sampledCluster]+=1
                        C[-1][i]=sampledCluster

                activeClusters=np.unique(C[-1])
                #print('activeclusters',activeClusters)
                nStar=len(activeClusters)
                for j in range(nStar):
                    y=self.data[C[-1]==activeClusters[j]] #current points of the cluster
                    n=y.shape[0]
                    #posterior parameters
                    mu0_n=(nu*mu0+np.sum(y))/(nu+n)
                    nu_n=nu+n
                    alpha_n=alpha+n/2
                    beta_n=beta+0.5*n*np.var(y)+n*nu/(2*(nu_n))*(np.mean(y)-mu0)**2
                    #Sample sigma2
                    sigma2[-1][activeClusters[j]]=ss.invgamma.rvs(a=alpha_n,scale=beta_n)
                    #Sample mu
                    mu[-1][activeClusters[j]]=ss.norm.rvs(loc=mu0_n,scale=np.sqrt(sigma2[-1][j]/nu_n))

                #Sample the concentration param following thanos augmentation method https://users.soe.ucsc.edu/~thanos/notes-2.pdf

                ##Sample eta, latent var
                eta.append(ss.beta.rvs(a=concParam[-1]+1,b=self.n))
                epsilon=(self.priorParamAlpha[0]+nStar-1)/(self.n*(self.priorParamAlpha[1]-np.log(eta[-1]))+self.priorParamAlpha[0]+nStar-1)
                if(np.random.random()<epsilon):
                    concParam.append(ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar, scale=1/(self.priorParamAlpha[1]-np.log(eta[-1]))))
                else:
                    concParam.append(ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar-1, scale=1/(self.priorParamAlpha[1]-np.log(eta[-1]))))

                N.append(1*N[-1])
                C.append(1*C[-1])
                sigma2.append(1*sigma2[-1][:])
                mu.append(1*mu[-1][:])
                numberActiveClusters.append(nStar)
            N.pop()
            C.pop()
            sigma2.pop()
            mu.pop()
            concParam.pop()
            del C[0:burnIn]
            del N[0:burnIn]
            del numberActiveClusters[0:burnIn]
            del mu[0:burnIn]
            del sigma2[0:burnIn]
            del concParam[0:burnIn]

            return N,C,sigma2,mu,concParam,numberActiveClusters

        else:
            sys.exit('This function is only designed for the conjugate prior model (Normal Inverse Gamma prior)')
    def nestedSampling(self, numIter, activeSetSize, GibbsStep,verbose=False):

        if self.priorDistLocScale=='Conjugate':
            mu=[]
            sigma2=[]
            alpha=[]
            L=np.zeros(activeSetSize) #llk of active set
            #Initialize the active sets with activeSetSize draws from the prior

            for i in range(activeSetSize):
                muTemp=np.zeros(self.n)
                sigma2Temp=np.ones(self.n)
                alpha.append(self.priorSimAlpha())
                for j in range(self.n):
                    if np.random.random()<alpha[-1]/(alpha[-1]+self.n-1): #Draw theta from base distribution
                        sigma2Temp[j]=ss.invgamma.rvs(a=self.priorParamLocScale[2],scale=self.priorParamLocScale[3])
                        muTemp[j]=ss.norm.rvs(loc=self.priorParamLocScale[0],scale=np.sqrt(sigma2Temp[j]/self.priorParamLocScale[1]))
                    else: #choose at random one of the other atoms
                        while True:
                            newIndex=np.random.choice(range(self.n))
                            if(newIndex!=j):
                                break
                        sigma2Temp[j]=sigma2Temp[newIndex]
                        muTemp[j]=muTemp[newIndex]
                    L[i]+=ss.norm.logpdf(self.data[j],loc=muTemp[j],scale=np.sqrt(sigma2Temp[j]))
                mu.append(muTemp)
                sigma2.append(sigma2Temp)
            #print(L,mu,sigma2)
            Z=0 #logevidence
            X=np.zeros(numIter) #prior volume
            llk=np.zeros(numIter) #log-likelihood chain
            #print(self.logLikelihood(mu[1],sigma2[1]))
            #print(L[1])

            for t in range(numIter):

                #Computing the prior volume
                X[t]=np.exp(-(t)/activeSetSize)


                minIndex=np.argmin(L)
                llk[t]=L[minIndex]
                #print(L[minIndex])
                #sample randomly one of the survivors for the init state of gibbs
                while True:
                    GibbsStart=int(np.random.choice(activeSetSize,1))
                    if GibbsStart!=minIndex:
                        break
                muGibbs=[]
                muGibbs.append(1*mu[GibbsStart])
                sigma2Gibbs=[]
                sigma2Gibbs.append(1*sigma2[GibbsStart])
                alphaGibbs=[]
                alphaGibbs.append(1*alpha[GibbsStart])

                for i in range(GibbsStep):

                    #currentLlk=self.logLikelihood(muGibbs[-1],sigma2Gibbs[-1])

                    #Update alpha where the proposal is the prior
                    alphaProposal=self.priorSimAlpha()
                    #acceptance ratio
                    ratioMH=0
                    for j in np.arange(1,self.n):
                        ratioMH+=np.log(alphaProposal/(alphaProposal+j)*ss.invgamma.pdf(sigma2Gibbs[-1][j],a=self.priorParamLocScale[2],scale=self.priorParamLocScale[3])*ss.norm.pdf(muGibbs[-1][j],loc=self.priorParamLocScale[0],scale=sigma2Gibbs[-1][j]/self.priorParamLocScale[1])+1/(alphaProposal+j)*np.sum(muGibbs[-1][range(j)]==muGibbs[-1][j]))-\
                            np.log(alphaGibbs[-1]/(alphaGibbs[-1]+j)*ss.invgamma.pdf(sigma2Gibbs[-1][j],a=self.priorParamLocScale[2],scale=self.priorParamLocScale[3])*ss.norm.pdf(muGibbs[-1][j],loc=self.priorParamLocScale[0],scale=sigma2Gibbs[-1][j]/self.priorParamLocScale[1])+1/(alphaGibbs[-1]+j)*np.sum(muGibbs[-1][range(j)]==muGibbs[-1][j]))
                    if np.log(np.random.random())<ratioMH and self.logLikelihood(muGibbs[-1],sigma2Gibbs[-1])>llk[t]:
                        alphaGibbs.append(alphaProposal)
                        #print(np.exp(ratioMH))
                    else:
                        alphaGibbs.append(1*alphaGibbs[-1])
                    if self.logLikelihood(muGibbs[-1],sigma2Gibbs[-1])<llk[t]:
                        print('false')


                    #simulate Theta from prior with a MH within gibbs step
                    currentLlk=self.logLikelihood(muGibbs[-1],sigma2Gibbs[-1])
                    for j in range(self.n):
                        if np.random.random()<alphaGibbs[-1]/(alphaGibbs[-1]+self.n-1):
                            sigma2Cand=ss.invgamma.rvs(a=self.priorParamLocScale[2],scale=self.priorParamLocScale[3])
                            muCand=ss.norm.rvs(loc=self.priorParamLocScale[0], scale=np.sqrt(sigma2Cand/self.priorParamLocScale[1]))
                        else: #choose at random one of the other atoms
                            while True:
                                newIndex=np.random.choice(range(self.n))
                                if(newIndex!=j):
                                    break
                            sigma2Cand=1*sigma2Gibbs[-1][newIndex]
                            muCand=1*muGibbs[-1][newIndex]
                        muTemp=1*muGibbs[-1]
                        sigma2Temp=1*sigma2Gibbs[-1]
                        muTemp[j]=muCand
                        sigma2Temp[j]=sigma2Cand
                        currentLlk=currentLlk-ss.norm.logpdf(self.data[j],loc=muGibbs[-1][j],scale=np.sqrt(sigma2Gibbs[-1][j]))+ss.norm.logpdf(self.data[j],loc=muCand,scale=np.sqrt(sigma2Cand))
                        #print(self.logLikelihood(muTemp,sigma2Temp))
                        #if self.logLikelihood(muTemp,sigma2Temp)>llk[t]:
                        if currentLlk>llk[t]:
                            #print('True')
                            muGibbs[-1][j]=muCand
                            sigma2Gibbs[-1][j]=sigma2Cand
                        else:
                            currentLlk=currentLlk+ss.norm.logpdf(self.data[j],loc=muGibbs[-1][j],scale=np.sqrt(sigma2Gibbs[-1][j]))-ss.norm.logpdf(self.data[j],loc=muCand,scale=np.sqrt(sigma2Cand))
                    muGibbs.append(muGibbs[-1])
                    sigma2Gibbs.append(sigma2Gibbs[-1])
                mu[minIndex]=1*muGibbs[-1]
                sigma2[minIndex]=1*sigma2Gibbs[-1]
                alpha[minIndex]=1*alphaGibbs[-1]

                L[minIndex]=self.logLikelihood(muGibbs[-1],sigma2Gibbs[-1])

                if(t>=2):
                    mm=np.max(np.log(-np.diff(X[np.arange(t)]))+llk[np.arange(t-1)+1])
                    Z=mm+np.log(np.sum(np.exp(np.log(-np.diff(X[np.arange(t)]))+llk[np.arange(t-1)+1]-mm)))
                    stopping=np.max(L)+np.log(X[t])
                    #print(llk)
                    if verbose==True:
                        print(Z,stopping,end="\r")
                    if stopping-Z<0.001:
                        if verbose==True:
                            print("Converged : ", Z, stopping-Z)
                        break
            if verbose==True:
                print("nombre max d'itérations atteint : ", t)
            return Z
        else:
            sys.exit('This function is only designed for the conjugate prior model (Normal Inverse Gamma prior)')

    def nestedSamplingAlloc(self,numIter,activeSetSize, GibbsStep,verbose=False):
        if self.priorDistLocScale=='Conjugate':
            #Itiniatilise allocations
            S=[]
            N=[]
            newClusterIndicator=[]
            alpha=[]
            L=np.zeros(activeSetSize) #logLlk of the active points

            for i in range(activeSetSize):
                S.append(np.zeros(self.n,dtype=int))
                N.append(np.zeros(self.n,dtype=int))
                newClusterIndicator.append(np.zeros(self.n,dtype=int))
                newClusterIndicator[-1][0]=1 #first obs assigned to a 'new' cluster
                N[-1][0]+=1 #first obs is assigned to first cluster
                alpha.append(self.priorSimAlpha())
                for j in range(1,self.n):
                    if np.log(np.random.random())<np.log(alpha[-1])-np.log(alpha[-1]+j): #Assign a new cluster :
                        emptyClusterIndex=next((idx for idx, val in np.ndenumerate(N[-1]) if val==0)) #Find the first zero cluster
                        N[-1][emptyClusterIndex]+=1
                        S[-1][j]=emptyClusterIndex[0]
                        newClusterIndicator[-1][j]=1
                    else : #assign existing cluster at random among the past allocations

                        newIndex=np.random.choice(self.n,p=N[-1]/j)
                        S[-1][j]=S[-1][newIndex]
                        N[-1][int(S[-1][newIndex])]+=1
                for s in np.unique(S[-1]):
                    mu0=self.priorParamLocScale[0]
                    nu=self.priorParamLocScale[1]
                    nu_n=nu+N[-1][s]
                    alphaParam=self.priorParamLocScale[2]
                    beta=self.priorParamLocScale[3]
                    y=self.data[S[-1]==s]
                    beta_n=beta+0.5*N[-1][s]*np.var(y)+N[-1][s]*nu/(2*(nu_n))*(np.mean(y)-mu0)**2
                    L[i]+=-0.5*N[-1][s]*np.log(2*np.pi)+0.5*np.log(nu/(nu+N[-1][s]))+alphaParam*np.log(beta/beta_n)-0.5*N[-1][s]*np.log(beta_n)+loggamma(alphaParam+0.5*N[-1][s])-loggamma(alphaParam)
            #print(S,N,alpha,L,newClusterIndicator)
            Z=0 #logevidence
            X=np.zeros(numIter) #prior volume
            llk=np.zeros(numIter) #log-likelihood chain


            for t in range(numIter):
                #Computing the prior volume
                X[t]=np.exp(-(t)/activeSetSize)


                minIndex=np.argmin(L)
                llk[t]=L[minIndex]
                while True:
                    GibbsStart=int(np.random.choice(activeSetSize,1))
                    if GibbsStart!=minIndex:
                        break
                alphaGibbs=[]
                newClusterIndicatorGibbs=[]
                SGibbs=[]
                NGibbs=[]

                alphaGibbs.append(1*alpha[GibbsStart])
                newClusterIndicatorGibbs.append(1*newClusterIndicator[GibbsStart])
                SGibbs.append(1*S[GibbsStart])
                NGibbs.append(1*N[GibbsStart])
                for u in range(GibbsStep):


                    NCand=1*NGibbs[-1]
                    SCand=1*SGibbs[-1]
                    for j in range(self.n):

                        #Sample a candidate for SCand[j] from the prior
                        NCand[SCand[j]]-=1
                        if np.random.random()<alphaGibbs[-1]/(self.n-1+alphaGibbs[-1]):
                            #print('hh')
                            emptyClusterIndex=next((idx for idx, val in np.ndenumerate(NCand) if val==0)) #Find the first zero cluster
                            NCand[emptyClusterIndex]+=1
                            SCand[j]=emptyClusterIndex[0]
                        else : #assign existing cluster wp prop to N
                            #print('ff')
                            #print(NCand)
                            #print(SCand)
                            clusterIndex=np.random.choice(self.n,p=NCand/(self.n-1))
                            #print(clusterIndex)
                            NCand[clusterIndex]+=1
                            SCand[j]=clusterIndex
                        if self.logLikelihoodAllocations(SCand,NCand)>llk[t]:
                            SGibbs[-1]=1*SCand
                            NGibbs[-1]=1*NCand
                        #print(NCand,SCand)
                    NGibbs.append(1*NGibbs[-1])
                    SGibbs.append(1*SGibbs[-1])

                    #Sample alpha
                    ##Sample eta, latent var
                    nStar=len(np.unique(SGibbs[-1]))
                    eta=ss.beta.rvs(a=alphaGibbs[-1]+1,b=self.n)
                    epsilon=(self.priorParamAlpha[0]+nStar-1)/(self.n*(self.priorParamAlpha[1]-np.log(eta))+self.priorParamAlpha[0]+nStar-1)
                    if(np.random.random()<epsilon):
                        alphaGibbs.append(ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar, scale=1/(self.priorParamAlpha[1]-np.log(eta))))
                    else:
                        alphaGibbs.append(ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar-1, scale=1/(self.priorParamAlpha[1]-np.log(eta))))


                    # #currentLlk=self.logLikelihood(muGibbs[-1],sigma2Gibbs[-1])

                    # #Update alpha where the proposal is the prior
                    # alphaProposal=self.priorSimAlpha()
                    # #acceptance ratio
                    # ratioMH=0
                    # for j in np.arange(0,self.n):
                    #     ratioMH+=np.log(newClusterIndicatorGibbs[-1][j]*alphaProposal/(alphaProposal+j)+(1-newClusterIndicatorGibbs[-1][j])/(alphaProposal+j))-\
                    #         np.log(newClusterIndicatorGibbs[-1][j]*alphaGibbs[-1]/(alphaGibbs[-1]+j)+(1-newClusterIndicatorGibbs[-1][j])/(alphaGibbs[-1]+j))
                    # if np.log(np.random.random())<ratioMH :
                    #     alphaGibbs.append(alphaProposal)
                    # else:
                    #     alphaGibbs.append(1*alphaGibbs[-1])


                    # #the sequence of cluster allocation is exchangeable



                    # for j in range(self.n):
                    #     NTemp=1*NGibbs[-1]
                    #     STemp=1*SGibbs[-1]
                    #     newClusterIndicatorTemp=1*newClusterIndicatorGibbs[-1]
                    #     if np.log(np.random.random())<np.log(alphaGibbs[-1])-np.log(alphaGibbs[-1]+j): #Assign a new cluster :
                    #         NTemp[STemp[j]]-=1
                    #         emptyClusterIndex=next((idx for idx, val in np.ndenumerate(NTemp) if val==0)) #Find the first zero cluster
                    #         NTemp[emptyClusterIndex]+=1
                    #         STemp[j]=emptyClusterIndex[0]
                    #         newClusterIndicatorTemp[j]=1
                    #     else: #choose at random one of the other atoms
                    #         NTemp[STemp[j]]-=1
                    #         while True:
                    #             newIndex=np.random.choice(self.n,p=NTemp/(self.n-1))
                    #             if(newIndex!=j):
                    #                 break
                    #         NTemp[STemp[newIndex]]+=1
                    #         STemp[j]=STemp[newIndex]
                    #         newClusterIndicatorTemp[j]=0
                    #     currentLlk=0
                    #     for s in np.unique(STemp):
                    #         mu0=self.priorParamLocScale[0]
                    #         nu=self.priorParamLocScale[1]
                    #         nu_n=nu+NTemp[s]
                    #         alphaParam=self.priorParamLocScale[2]
                    #         beta=self.priorParamLocScale[3]
                    #         y=self.data[STemp==s]
                    #         beta_n=beta+0.5*NTemp[s]*np.var(y)+NTemp[s]*nu/(2*(nu_n))*(np.mean(y)-mu0)**2
                    #         currentLlk+=-0.5*NTemp[s]*np.log(2*np.pi)+0.5*np.log(nu/(nu+NTemp[s]))+alphaParam*np.log(beta/beta_n)-0.5*NTemp[s]*np.log(beta_n)+loggamma(alphaParam+0.5*NTemp[s])-loggamma(alphaParam)
                    #     if currentLlk>llk[t]:
                    #         SGibbs.append(STemp)
                    #         NGibbs.append(NTemp)
                    #         newClusterIndicatorGibbs.append(newClusterIndicatorTemp)

                S[minIndex]=1*SGibbs[-1]
                N[minIndex]=1*NGibbs[-1]
                newClusterIndicator[minIndex]=1*newClusterIndicatorGibbs[-1]
                alpha[minIndex]=1*alphaGibbs[-1]

                L[minIndex]=0
                for s in np.unique(S[-1]):
                    mu0=self.priorParamLocScale[0]
                    nu=self.priorParamLocScale[1]
                    nu_n=nu+NGibbs[-1][s]
                    alphaParam=self.priorParamLocScale[2]
                    beta=self.priorParamLocScale[3]
                    y=self.data[S[-1]==s]
                    beta_n=beta+0.5*NGibbs[-1][s]*np.var(y)+NGibbs[-1][s]*nu/(2*(nu_n))*(np.mean(y)-mu0)**2
                    L[minIndex]+=-0.5*NGibbs[-1][s]*np.log(2*np.pi)+0.5*np.log(nu/(nu+NGibbs[-1][s]))+alphaParam*np.log(beta/beta_n)-0.5*NGibbs[-1][s]*np.log(beta_n)+loggamma(alphaParam+0.5*NGibbs[-1][s])-loggamma(alphaParam)

                if(t>=2):
                    mm=np.max(np.log(-np.diff(X[np.arange(t)]))+llk[np.arange(t-1)+1])
                    Z=mm+np.log(np.sum(np.exp(np.log(-np.diff(X[np.arange(t)]))+llk[np.arange(t-1)+1]-mm)))
                    stopping=np.max(L)+np.log(X[t])

                    #stopping=np.max(L)
                    #print(llk)
                    if verbose==True:
                        print(Z,stopping,np.log(X[t]),np.max(L),end="\r")
                    if stopping-Z<0.001:
                        if verbose==True:
                            print("Converged : ", Z, stopping-Z)
                        break
            if verbose==True:
                print("nombre max d'itérations atteint : ", t)
            return Z

        return 0

    def harmonicMean(self,numIterGibbs,burnIn):
        N,S,sigma2,mu,alpha,_=self.NealsGibbsSamplerConjugate(numIterGibbs,burnIn)
        logInvLlk=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            for i in range(self.n):
                logInvLlk[t]-=ss.norm.logpdf(self.data[i],loc=mu[t][int(S[t][i])],scale=np.sqrt(sigma2[t][int(S[t][i])]))
        return -(np.log(1/(numIterGibbs-burnIn))+logsumexp(logInvLlk))
    def arithmeticMean(self,numSim):
        alpha=[]
        S=[]
        N=[]
        newClusterIndicator=[]
        logLlk=np.zeros(numSim)
        for t in range(numSim):
            alpha.append(self.priorSimAlpha())
            S.append(np.zeros(self.n,dtype=int))
            N.append(np.zeros(self.n,dtype=int))
            newClusterIndicator.append(np.zeros(self.n,dtype=int))
            newClusterIndicator[-1][0]=1 #first obs assigned to a 'new' cluster
            N[-1][0]+=1 #first obs is assigned to first cluster
            for i in range(1,self.n):
                if np.log(np.random.random())<np.log(alpha[-1])-np.log(alpha[-1]+i): #Assign a new cluster :
                    emptyClusterIndex=next((idx for idx, val in np.ndenumerate(N[-1]) if val==0)) #Find the first zero cluster
                    N[-1][emptyClusterIndex]+=1
                    S[-1][i]=emptyClusterIndex[0]
                    newClusterIndicator[-1][i]=1
                else : #assign existing cluster at random among the past allocations
                    newIndex=np.random.choice(range(i))
                    S[-1][i]=S[-1][newIndex]
                    N[-1][int(S[-1][newIndex])]+=1
            for s in np.unique(S[-1]):
                mu0=self.priorParamLocScale[0]
                nu=self.priorParamLocScale[1]
                nu_n=nu+N[-1][s]
                alphaParam=self.priorParamLocScale[2]
                beta=self.priorParamLocScale[3]
                y=self.data[S[-1]==s]
                beta_n=beta+0.5*N[-1][s]*np.var(y)+N[-1][s]*nu/(2*(nu_n))*(np.mean(y)-mu0)**2
                logLlk[t]+=-0.5*N[-1][s]*np.log(2*np.pi)+0.5*np.log(nu/(nu+N[-1][s]))+alphaParam*np.log(beta/beta_n)-0.5*N[-1][s]*np.log(beta_n)+loggamma(alphaParam+0.5*N[-1][s])-loggamma(alphaParam)

        return -np.log(numSim)+logsumexp(logLlk)
    def GibbsSamplingAlloc(self, numIter,burnIn):
        if self.priorDistLocScale=='Conjugate':



            C=[]
            N=[]
            concParam=[] #concentration parameter of the DP
            eta=[]#latent var for sampling concPram of DP
            concParam.append(self.priorSimAlpha())#initialise alpha with a draw from the prior
            C.append(np.zeros(self.n,dtype=int)) #Assign all the observation to cluster 0
            N.append(np.zeros(self.n,dtype=int))#Variable that counts the number of obs per cluster
            N[0][0]=self.n #all the obs are in the first cluster a prioro

            numberActiveClusters=[] #number of active clusters
            for t in range(numIter):
                print(np.round(t/numIter*100),'%',end="\r")
                for i in range(self.n):


                    #Draw a new value for c_i
                    N[-1][C[-1][i]]+=-1 #Remove the previous allocation of y_i

                    activeClusters=np.unique(C[-1][np.arange(len(C[-1]))!=i])
                    C[-1][i]=self.n+1 #temporary remove data i alloc by assigning it an impossible cluster
                    nStar=len(activeClusters)
                    samplingWeights=np.zeros(nStar+1) #log weights to sample c_i
                    #print(nStar)
                    for j in range(nStar): #iterate through all the active clusters
                        n_c=N[-1][activeClusters[j]] #number of obs in cluster c
                        mu0=self.priorParamLocScale[0]
                        nu=self.priorParamLocScale[1]
                        alpha=self.priorParamLocScale[2]
                        beta=self.priorParamLocScale[3]
                        nu_nc=nu+n_c
                        alpha_nc=alpha+n_c/2
                        y_c=self.data[C[-1]==activeClusters[j]]
                        mu0_nc=(np.sum(y_c)+nu*mu0)/(n_c+nu)
                        beta_nc=beta+0.5*n_c*np.var(y_c)+nu*n_c/(2*nu_nc)*(np.mean(y_c)-mu0)**2
                        beta_prime=beta_nc+nu_nc/(2*(nu_nc+1))*(self.data[i]-mu0_nc)**2
                        samplingWeights[j]=np.log(n_c/(self.n-1+concParam[-1]))-0.5*np.log(2*np.pi*beta_prime)+0.5*np.log(nu_nc/(nu_nc+1))+alpha_nc*np.log(beta_nc/beta_prime)+loggamma(alpha_nc+0.5)-loggamma(alpha_nc)

                    nu=self.priorParamLocScale[1]
                    alpha=self.priorParamLocScale[2]
                    beta=self.priorParamLocScale[3]
                    mu0=self.priorParamLocScale[0]
                    beta_n=beta+nu/(2*(nu+1))*(self.data[i]-mu0)**2
                    samplingWeights[nStar]=np.log(concParam[-1]/(self.n-1+concParam[-1]))-0.5*np.log(2*np.pi*beta_n)+alpha*np.log(beta/beta_n)+0.5*np.log(nu/(nu+1))+loggamma(alpha+0.5)-loggamma(alpha)
                    #print(samplingWeights)
                    sampledIndex=catDistLogProb(samplingWeights)
                    if sampledIndex==nStar: #create a new cluster
                        emptyClusterIndex=next((idx for idx, val in np.ndenumerate(N[-1]) if val==0)) #Find the first zero cluster
                        #print('add cluster ',emptyClusterIndex)
                        N[-1][emptyClusterIndex]+=1
                        C[-1][i]=emptyClusterIndex[0]
                        #print('active cluster ', np.unique(C[-1]))


                    else:
                        sampledCluster=activeClusters[sampledIndex]
                        N[-1][sampledCluster]+=1
                        C[-1][i]=sampledCluster
                #Sample the concentration param following thanos augmentation method https://users.soe.ucsc.edu/~thanos/notes-2.pdf

                ##Sample eta, latent var
                eta.append(ss.beta.rvs(a=concParam[-1]+1,b=self.n))
                epsilon=(self.priorParamAlpha[0]+nStar-1)/(self.n*(self.priorParamAlpha[1]-np.log(eta[-1]))+self.priorParamAlpha[0]+nStar-1)
                if(np.random.random()<epsilon):
                    concParam.append(ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar, scale=1/(self.priorParamAlpha[1]-np.log(eta[-1]))))
                else:
                    concParam.append(ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar-1, scale=1/(self.priorParamAlpha[1]-np.log(eta[-1]))))

                N.append(1*N[-1])
                C.append(1*C[-1])
                #print(N[-1])

                numberActiveClusters.append(len(np.unique(C[-1])))
            N.pop()
            C.pop()
            concParam.pop()
            del C[0:burnIn]
            del N[0:burnIn]
            del numberActiveClusters[0:burnIn]
            del concParam[0:burnIn]

            return N,C,concParam,numberActiveClusters

        else:
            sys.exit('This function is only designed for the conjugate prior model (Normal Inverse Gamma prior)')

    def ChibEstimator(self,numIterGibbs,burnIn):


        #Posterior ordinate estimation
        N,C,concParam,nStar=self.GibbsSamplingAlloc(numIterGibbs,burnIn)

        alphaStar=np.median(concParam)
        print(alphaStar)
        #alphaStar=10
        eta=ss.beta.rvs(a=alphaStar+1,b=self.n,size=numIterGibbs-burnIn)
        logPosteriorAlpha=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
           p=(self.priorParamAlpha[0]+nStar[t]-1)/(self.n*(self.priorParamAlpha[1]-np.log(eta[t]))+self.priorParamAlpha[0]+nStar[t]-1)
           logPosteriorAlpha[t]=np.log(p*ss.gamma.pdf(alphaStar,a=self.priorParamAlpha[0]+nStar[t], scale=1/(self.priorParamAlpha[1]-np.log(eta[t])))+(1-p)*ss.gamma.pdf(alphaStar,a=self.priorParamAlpha[0]+nStar[t]-1, scale=1/(self.priorParamAlpha[1]-np.log(eta[t]))))
        logPosteriorAlpha=-np.log(numIterGibbs-burnIn)+logsumexp(logPosteriorAlpha)

        #logLikelihood ordinate following gibbs noation
        w=np.zeros(numIterGibbs-burnIn)
        for t in range(numIterGibbs-burnIn):
            #Step 1
            u=np.zeros(self.n)
            s=-np.ones(self.n,dtype=int) #set all cluster allocations to -1
            n=np.zeros(self.n,dtype=int)
            beta_n=self.priorParamLocScale[3]+self.priorParamLocScale[1]/(2*(self.priorParamLocScale[1]+1))*(self.data[0]-self.priorParamLocScale[0])**2
            u[0]=-0.5*np.log(2*np.pi*beta_n)+0.5*np.log(self.priorParamLocScale[1]/(self.priorParamLocScale[1]+1))+self.priorParamLocScale[2]*np.log(self.priorParamLocScale[3]/beta_n)+loggamma(self.priorParamLocScale[2]+0.5)-loggamma(self.priorParamLocScale[2])
            s[0]=0
            n[0]+=1
            for i in range(1,self.n):
                activeClusters=np.unique(s[:i])
                nStar=len(activeClusters)
                samplingWeights=np.zeros(nStar+1)

                for j in range(nStar): #iterate through all the active clusters
                    n_c=n[activeClusters[j]] #number of obs in cluster c
                    mu0=self.priorParamLocScale[0]
                    nu=self.priorParamLocScale[1]
                    alpha=self.priorParamLocScale[2]
                    beta=self.priorParamLocScale[3]
                    nu_nc=nu+n_c
                    alpha_nc=alpha+n_c/2
                    y_c=self.data[s==activeClusters[j]]
                    mu0_nc=(np.sum(y_c)+nu*mu0)/(n_c+nu)
                    beta_nc=beta+0.5*n_c*np.var(y_c)+nu*n_c/(2*nu_nc)*(np.mean(y_c)-mu0)**2
                    beta_prime=beta_nc+nu_nc/(2*(nu_nc+1))*(self.data[i]-mu0_nc)**2
                    samplingWeights[j]=np.log(n_c/(i+alphaStar))-0.5*np.log(2*np.pi*beta_prime)+0.5*np.log(nu_nc/(nu_nc+1))+alpha_nc*np.log(beta_nc/beta_prime)+loggamma(alpha_nc+0.5)-loggamma(alpha_nc)

                nu=self.priorParamLocScale[1]
                alpha=self.priorParamLocScale[2]
                beta=self.priorParamLocScale[3]
                mu0=self.priorParamLocScale[0]
                beta_n=beta+nu/(2*(nu+1))*(self.data[i]-mu0)**2
                samplingWeights[nStar]=np.log(alphaStar/(i+alphaStar))-0.5*np.log(2*np.pi*beta_n)+alpha*np.log(beta/beta_n)+0.5*np.log(nu/(nu+1))+loggamma(alpha+0.5)-loggamma(alpha)


                u[i]=logsumexp(samplingWeights)

                #Step 2
                sampledIndex=catDistLogProb(samplingWeights)
                if sampledIndex==nStar: #create a new cluster
                    emptyClusterIndex=next((idx for idx, val in np.ndenumerate(n) if val==0)) #Find the first zero cluster
                    #print('add cluster ',emptyClusterIndex)
                    s[i]=emptyClusterIndex[0]
                    n[emptyClusterIndex]+=1
                else:
                    sampledCluster=activeClusters[sampledIndex]
                    n[sampledCluster]+=1
                    s[i]=sampledCluster
            w[t]=np.sum(u)
        logLikelihoodOrdinate=-np.log(numIterGibbs-burnIn)+logsumexp(w)

        logPriorOrdinate=ss.gamma.logpdf(alphaStar,a=self.priorParamAlpha[0],scale=1/self.priorParamAlpha[1])
        print('llk : ',logLikelihoodOrdinate)
        print('prior : ', logPriorOrdinate)
        print('posterior : ',logPosteriorAlpha)
        return logLikelihoodOrdinate+logPriorOrdinate-logPosteriorAlpha
    def logLikelihoodAllocations(self,S,N): ###Attention peut etre qu'il manque un facteur 2 devant le pi...
        clusters=np.unique(S)
        llk=0
        for k in range(len(clusters)):
            mu0=self.priorParamLocScale[0]
            nu=self.priorParamLocScale[1]
            alphaParam=self.priorParamLocScale[2]
            beta=self.priorParamLocScale[3]
            n=N[clusters[k]]
            y=self.data[S==clusters[k]]
            beta_n=beta+0.5*n*np.var(y)+n*nu/(2*(nu+n))*(np.mean(y)-mu0)**2
            llk+=-n*0.5*np.log(np.pi*beta_n)+0.5*np.log(nu/(nu+n))+alphaParam*np.log(beta/beta_n)+loggamma(alphaParam+n/2)-loggamma(alphaParam)
        return llk
    def SMCAlloc(self,numParticles,GibbsStep,T,ESSthreshold): #####ATTENTION ETAPE RESAMPLING ETRANGE

        if self.priorDistLocScale=='Conjugate':
            ESSthreshold=ESSthreshold*numParticles
            #Initialise with numParticles particles drawn from the prior
            S=[]
            N=[]
            alpha=[]
            W=-np.log(numParticles)*np.ones(numParticles) #logLlk of the active points
            temp=[]
            temp.append(0)
            #First step : sample from prior
            for i in range(numParticles):
                S.append(np.zeros(self.n,dtype=int))
                N.append(np.zeros(self.n,dtype=int))
                N[-1][0]+=1 #first obs is assigned to first cluster
                alpha.append(self.priorSimAlpha())
                for j in range(1,self.n):
                    if np.log(np.random.random())<np.log(alpha[-1])-np.log(alpha[-1]+j): #Assign a new cluster :
                        emptyClusterIndex=next((idx for idx, val in np.ndenumerate(N[-1]) if val==0)) #Find the first zero cluster
                        N[-1][emptyClusterIndex]+=1
                        S[-1][j]=emptyClusterIndex[0]
                    else : #assign existing cluster at random among the past allocations
                        newIndex=np.random.choice(self.n,p=N[-1]/j)
                        S[-1][j]=newIndex
                        N[-1][newIndex]+=1

            t=0
            Z=0
            while True:

                if t/T<0.20:
                    temp.append(temp[t]+1/T)
                    #print(temp[t])
                elif t/T<0.40 and temp[t-1]<0.4-2/T:
                    temp.append(temp[t]+2/T)
                    #print(temp[t])
                elif t/T<1 and temp[t-1]<1-3/T and temp[t]+3/T<1:
                    temp.append(temp[t]+3/T)
                    #print(temp[t])
                else:
                    temp.append(1)
                    #print(temp[t])
                print(temp[-1])
                for i in range(numParticles):

                    #Update weights

                    W[i]=W[i]+(temp[-1]-temp[-2])*self.logLikelihoodAllocations(S[i],N[i]) #log weight update
                Z+=logsumexp(W)
                #Normalise weights
                W=W-logsumexp(W)

                #Compute ESS
                ESS=1/(logsumexp(2*W))

                if ESS<ESSthreshold : #Resample
                    for i in range(numParticles):
                        sampIndex=catDistLogProb(W)
                        S[i]=S[sampIndex]
                        N[i]=N[sampIndex]
                    #The weights are then equal
                    W=-np.log(numParticles)*np.ones(numParticles)

                #Move with MH within Gibbs

                for i in range(numParticles):
                    NCand=1*N[i]
                    SCand=1*S[i]
                    for j in range(self.n):

                        #Sample a candidate for SCand[j] from the prior
                        NCand[SCand[j]]-=1
                        if np.random.random()<alpha[-1]/(self.n-1+alpha[-1]):
                            #print('hh')
                            emptyClusterIndex=next((idx for idx, val in np.ndenumerate(NCand) if val==0)) #Find the first zero cluster
                            NCand[emptyClusterIndex]+=1
                            SCand[j]=emptyClusterIndex[0]
                        else : #assign existing cluster wp prop to N
                            #print('ff')
                            #print(NCand)
                            #print(SCand)
                            clusterIndex=np.random.choice(self.n,p=NCand/(self.n-1))
                            #print(clusterIndex)
                            NCand[clusterIndex]+=1
                            SCand[j]=clusterIndex
                        if np.log(np.random.random())<temp[-1]*(self.logLikelihoodAllocations(SCand,NCand)-self.logLikelihoodAllocations(S[i],N[i])):
                            #print('treu')
                            S[i]=np.array(SCand)
                            N[i]=np.array(NCand)
                        #print(NCand,SCand)

                    #Sample alpha
                    ##Sample eta, latent var
                    nStar=len(np.unique(S[i]))
                    eta=ss.beta.rvs(a=alpha[i]+1,b=self.n)
                    epsilon=(self.priorParamAlpha[0]+nStar-1)/(self.n*(self.priorParamAlpha[1]-np.log(eta))+self.priorParamAlpha[0]+nStar-1)
                    if(np.random.random()<epsilon):
                        alpha[i]=ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar, scale=1/(self.priorParamAlpha[1]-np.log(eta)))
                    else:
                        alpha[i]=ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar-1, scale=1/(self.priorParamAlpha[1]-np.log(eta)))




                t+=1
                if (temp[-1]==1):
                    break


        return Z


    def SMCAllocWithGibbsMoveStep(self,numParticles,GibbsStep,T,ESSthreshold):

        if self.priorDistLocScale=='Conjugate':
            ESSthreshold=ESSthreshold*numParticles
            #Initialise with numParticles particles drawn from the prior
            S=[]
            N=[]
            concParam=[]
            W=-np.log(numParticles)*np.ones(numParticles) #logLlk of the active points
            temp=[]
            temp.append(0)
            #First step : sample from prior
            for i in range(numParticles):
                S.append(np.zeros(self.n,dtype=int))
                N.append(np.zeros(self.n,dtype=int))
                N[-1][0]+=1 #first obs is assigned to first cluster
                concParam.append(self.priorSimAlpha())
                for j in range(1,self.n):
                    if np.log(np.random.random())<np.log(concParam[-1])-np.log(concParam[-1]+j): #Assign a new cluster :
                        emptyClusterIndex=next((idx for idx, val in np.ndenumerate(N[-1]) if val==0)) #Find the first zero cluster
                        N[-1][emptyClusterIndex]+=1
                        S[-1][j]=emptyClusterIndex[0]
                    else : #assign existing cluster at random among the past allocations
                        newIndex=np.random.choice(self.n,p=N[-1]/j)
                        S[-1][j]=newIndex
                        N[-1][newIndex]+=1

            t=0
            Z=0
            while True:

                if t/T<0.20:
                    temp.append(temp[t]+1/T)
                    #print(temp[t])
                elif t/T<0.40 and temp[t-1]<0.4-2/T:
                    temp.append(temp[t]+2/T)
                    #print(temp[t])
                elif t/T<1 and temp[t-1]<1-3/T and temp[t]+3/T<1:
                    temp.append(temp[t]+3/T)
                    #print(temp[t])
                else:
                    temp.append(1)
                    #print(temp[t])
                print(temp[-1])
                for i in range(numParticles):

                    #Update weights

                    W[i]=W[i]+(temp[-1]-temp[-2])*self.logLikelihoodAllocations(S[i],N[i]) #log weight update
                Z+=logsumexp(W)
                #Normalise weights
                W=W-logsumexp(W)

                #Compute ESS
                ESS=1/(logsumexp(2*W))
                print('ESS:',ESS)
                if ESS<ESSthreshold : #Resample
                    for i in range(numParticles):
                        sampIndex=catDistLogProb(W)
                        S[i]=S[sampIndex]
                        N[i]=N[sampIndex]
                    #The weights are then equal
                    W=-np.log(numParticles)*np.ones(numParticles)

                #Move with MH within Gibbs
                mu0=self.priorParamLocScale[0]
                nu=self.priorParamLocScale[1]
                alpha=self.priorParamLocScale[2]
                beta=self.priorParamLocScale[3]
                for i in range(numParticles):
                    NCand=1*N[i]
                    SCand=1*S[i]
                    for j in range(self.n):

                        #Draw a new value for S_i
                        N[i][S[i][j]]+=-1 #Remove the previous allocation of y_i

                        activeClusters=np.unique(S[i][np.arange(len(S[i]))!=j])
                        S[i][j]=self.n+1 #temporary remove data i alloc by assigning it an impossible cluster
                        nStar=len(activeClusters)
                        samplingWeights=np.zeros(nStar+1) #log weights to sample s_i
                        #print(nStar)
                        for k in range(nStar): #iterate through all the active clusters
                            n_c=N[i][activeClusters[k]] #number of obs in cluster c
                            nu_nc=nu+temp[-1]*n_c
                            alpha_nc=alpha+(temp[-1]*n_c/2)
                            y_c=self.data[S[i]==activeClusters[k]]
                            mu0_nc=(temp[-1]*np.sum(y_c)+nu*mu0)/nu_nc
                            beta_nc=beta+0.5*n_c*temp[-1]*np.var(y_c)+temp[-1]*nu*n_c/(2*nu_nc)*(np.mean(y_c)-mu0)**2
                            beta_prime=beta_nc+(temp[-1]*nu_nc)/(2*(nu_nc+temp[-1]))*(self.data[j]-mu0_nc)**2
                            samplingWeights[k]=np.log(n_c/(self.n-1+concParam[i]))-(temp[-1]/2)*np.log(2*np.pi*beta_prime)+0.5*np.log(nu_nc/(nu_nc+temp[-1]))+alpha_nc*np.log(beta_nc/beta_prime)+loggamma(alpha_nc+temp[-1]/2)-loggamma(alpha_nc)

                        beta_n=beta+(temp[-1]*nu)/(2*(nu+temp[-1]))*(self.data[j]-mu0)**2
                        samplingWeights[nStar]=np.log(concParam[i]/(self.n-1+concParam[i]))-(temp[-1]/2)*np.log(2*np.pi*beta_n)+alpha*np.log(beta/beta_n)+0.5*np.log(nu/(nu+temp[-1]))+loggamma(alpha+temp[-1]/2)-loggamma(alpha)

                        sampledIndex=catDistLogProb(samplingWeights)
                        if sampledIndex==nStar: #create a new cluster
                            emptyClusterIndex=next((idx for idx, val in np.ndenumerate(N[i]) if val==0)) #Find the first zero cluster
                            #print('add cluster ',emptyClusterIndex)
                            N[i][emptyClusterIndex]+=1
                            S[i][j]=emptyClusterIndex[0]
                            #print('active cluster ', np.unique(C[-1]))


                        else:
                            sampledCluster=activeClusters[sampledIndex]
                            N[i][sampledCluster]+=1
                            S[i][j]=sampledCluster
                    #Sample the concentration param following thanos augmentation method https://users.soe.ucsc.edu/~thanos/notes-2.pdf

                    ##Sample eta, latent var
                    nStar=len(np.unique(S[i]))
                    eta=ss.beta.rvs(a=concParam[i]+1,b=self.n)
                    epsilon=(self.priorParamAlpha[0]+nStar-1)/(self.n*(self.priorParamAlpha[1]-np.log(eta))+self.priorParamAlpha[0]+nStar-1)
                    if(np.random.random()<epsilon):
                        concParam[i]=ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar, scale=1/(self.priorParamAlpha[1]-np.log(eta)))
                    else:
                        concParam[i]=ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar-1, scale=1/(self.priorParamAlpha[1]-np.log(eta)))





                t+=1
                if (temp[-1]==1):
                    break


        return Z

    def ChibEstimatorFixedConcParam(self,numIter,concParam):


        # #Posterior ordinate estimation
        # N,C,concParam,nStar=self.GibbsSamplingAlloc(numIterGibbs,burnIn)
        #
        # alphaStar=np.median(concParam)
        # print(alphaStar)
        # #alphaStar=10
        # eta=ss.beta.rvs(a=alphaStar+1,b=self.n,size=numIterGibbs-burnIn)
        # logPosteriorAlpha=np.zeros(numIterGibbs-burnIn)
        # for t in range(numIterGibbs-burnIn):
        #    p=(self.priorParamAlpha[0]+nStar[t]-1)/(self.n*(self.priorParamAlpha[1]-np.log(eta[t]))+self.priorParamAlpha[0]+nStar[t]-1)
        #    logPosteriorAlpha[t]=np.log(p*ss.gamma.pdf(alphaStar,a=self.priorParamAlpha[0]+nStar[t], scale=1/(self.priorParamAlpha[1]-np.log(eta[t])))+(1-p)*ss.gamma.pdf(alphaStar,a=self.priorParamAlpha[0]+nStar[t]-1, scale=1/(self.priorParamAlpha[1]-np.log(eta[t]))))
        # logPosteriorAlpha=-np.log(numIterGibbs-burnIn)+logsumexp(logPosteriorAlpha)

        #logLikelihood ordinate following gibbs noation
        w=np.zeros(numIter)
        for t in range(numIter):
            #Step 1
            u=np.zeros(self.n)
            s=-np.ones(self.n,dtype=int) #set all cluster allocations to -1
            n=np.zeros(self.n,dtype=int)
            beta_n=self.priorParamLocScale[3]+self.priorParamLocScale[1]/(2*(self.priorParamLocScale[1]+1))*(self.data[0]-self.priorParamLocScale[0])**2
            u[0]=-0.5*np.log(2*np.pi*beta_n)+0.5*np.log(self.priorParamLocScale[1]/(self.priorParamLocScale[1]+1))+self.priorParamLocScale[2]*np.log(self.priorParamLocScale[3]/beta_n)+loggamma(self.priorParamLocScale[2]+0.5)-loggamma(self.priorParamLocScale[2])
            s[0]=0
            n[0]+=1
            for i in range(1,self.n):
                activeClusters=np.unique(s[:i])
                nStar=len(activeClusters)
                samplingWeights=np.zeros(nStar+1)

                for j in range(nStar): #iterate through all the active clusters
                    n_c=n[activeClusters[j]] #number of obs in cluster c
                    mu0=self.priorParamLocScale[0]
                    nu=self.priorParamLocScale[1]
                    alpha=self.priorParamLocScale[2]
                    beta=self.priorParamLocScale[3]
                    nu_nc=nu+n_c
                    alpha_nc=alpha+n_c/2
                    y_c=self.data[s==activeClusters[j]]
                    mu0_nc=(np.sum(y_c)+nu*mu0)/(n_c+nu)
                    beta_nc=beta+0.5*n_c*np.var(y_c)+nu*n_c/(2*nu_nc)*(np.mean(y_c)-mu0)**2
                    beta_prime=beta_nc+nu_nc/(2*(nu_nc+1))*(self.data[i]-mu0_nc)**2
                    samplingWeights[j]=np.log(n_c/(i+concParam))-0.5*np.log(2*np.pi*beta_prime)+0.5*np.log(nu_nc/(nu_nc+1))+alpha_nc*np.log(beta_nc/beta_prime)+loggamma(alpha_nc+0.5)-loggamma(alpha_nc)

                nu=self.priorParamLocScale[1]
                alpha=self.priorParamLocScale[2]
                beta=self.priorParamLocScale[3]
                mu0=self.priorParamLocScale[0]
                beta_n=beta+nu/(2*(nu+1))*(self.data[i]-mu0)**2
                samplingWeights[nStar]=np.log(concParam/(i+concParam))-0.5*np.log(2*np.pi*beta_n)+alpha*np.log(beta/beta_n)+0.5*np.log(nu/(nu+1))+loggamma(alpha+0.5)-loggamma(alpha)


                u[i]=logsumexp(samplingWeights)

                #Step 2
                sampledIndex=catDistLogProb(samplingWeights)
                if sampledIndex==nStar: #create a new cluster
                    emptyClusterIndex=next((idx for idx, val in np.ndenumerate(n) if val==0)) #Find the first zero cluster
                    #print('add cluster ',emptyClusterIndex)
                    s[i]=emptyClusterIndex[0]
                    n[emptyClusterIndex]+=1
                else:
                    sampledCluster=activeClusters[sampledIndex]
                    n[sampledCluster]+=1
                    s[i]=sampledCluster
            w[t]=np.sum(u)
        logLikelihoodOrdinate=-np.log(numIter)+logsumexp(w)

        #logPriorOrdinate=ss.gamma.logpdf(concParam,a=self.priorParamAlpha[0],scale=1/self.priorParamAlpha[1])
        print('llk : ',logLikelihoodOrdinate)
        #print('prior : ', logPriorOrdinate)
        #print('posterior : ',logPosteriorAlpha)
        return logLikelihoodOrdinate

    def SMCAllocFixedConcParamWithGibbsMoveStep(self,numParticles,GibbsStep,T,ESSthreshold,concParam):

        if self.priorDistLocScale=='Conjugate':
            ESSthreshold=ESSthreshold*numParticles
            #Initialise with numParticles particles drawn from the prior
            S=[]
            N=[]

            W=-np.log(numParticles)*np.ones(numParticles) #logLlk of the active points
            temp=[]
            temp.append(0)
            #First step : sample from prior
            for i in range(numParticles):
                S.append(np.zeros(self.n,dtype=int))
                N.append(np.zeros(self.n,dtype=int))
                N[-1][0]+=1 #first obs is assigned to first cluster

                for j in range(1,self.n):
                    if np.log(np.random.random())<np.log(concParam)-np.log(concParam+j): #Assign a new cluster :
                        emptyClusterIndex=next((idx for idx, val in np.ndenumerate(N[-1]) if val==0)) #Find the first zero cluster
                        N[-1][emptyClusterIndex]+=1
                        S[-1][j]=emptyClusterIndex[0]
                    else : #assign existing cluster at random among the past allocations
                        newIndex=np.random.choice(self.n,p=N[-1]/j)
                        S[-1][j]=newIndex
                        N[-1][newIndex]+=1
                #W[i]=self.logLikelihoodAllocations(S[i],N[i])
            t=0
            Z=0
            while True:

                if t/T<0.20:
                    temp.append(temp[t]+1/T)
                    #print(temp[t])
                elif t/T<0.40 and temp[t-1]<0.4-2/T:
                    temp.append(temp[t]+2/T)
                    #print(temp[t])
                elif t/T<1 and temp[t-1]<1-3/T and temp[t]+3/T<1:
                    temp.append(temp[t]+3/T)
                    #print(temp[t])
                else:
                    temp.append(1)
                    #print(temp[t])
                print(temp[-1])
                #print(temp[-1]-temp[-2])
                for i in range(numParticles):

                    #Update weights

                    W[i]=(temp[-1]-temp[-2])*self.logLikelihoodAllocations(S[i],N[i]) #log weight update
                Z+=-np.log(numParticles)+logsumexp(W)
                #Normalise weights
                W=W-logsumexp(W)

                #Compute ESS
                ESS=1/(logsumexp(2*W))

                if ESS<ESSthreshold : #Resample
                    for i in range(numParticles):
                        sampIndex=catDistLogProb(W)
                        S[i]=S[sampIndex]
                        N[i]=N[sampIndex]
                    #The weights are then equal
                    W=-np.log(numParticles)*np.ones(numParticles)

                #Move with MH within Gibbs
                mu0=self.priorParamLocScale[0]
                nu=self.priorParamLocScale[1]
                alpha=self.priorParamLocScale[2]
                beta=self.priorParamLocScale[3]
                for i in range(numParticles):
                    NCand=1*N[i]
                    SCand=1*S[i]
                    for j in range(self.n):

                        #Draw a new value for S_i
                        N[i][S[i][j]]+=-1 #Remove the previous allocation of y_i

                        activeClusters=np.unique(S[i][np.arange(len(S[i]))!=j])
                        S[i][j]=self.n+1 #temporary remove data i alloc by assigning it an impossible cluster
                        nStar=len(activeClusters)
                        samplingWeights=np.zeros(nStar+1) #log weights to sample s_i
                        #print(nStar)
                        for k in range(nStar): #iterate through all the active clusters
                            n_c=N[i][activeClusters[k]] #number of obs in cluster c
                            nu_nc=nu+temp[-1]*n_c
                            alpha_nc=alpha+(temp[-1]*n_c/2)
                            y_c=self.data[S[i]==activeClusters[k]]
                            mu0_nc=(temp[-1]*np.sum(y_c)+nu*mu0)/nu_nc
                            beta_nc=beta+0.5*n_c*temp[-1]*np.var(y_c)+temp[-1]*nu*n_c/(2*nu_nc)*(np.mean(y_c)-mu0)**2
                            beta_prime=beta_nc+(temp[-1]*nu_nc)/(2*(nu_nc+temp[-1]))*(self.data[j]-mu0_nc)**2
                            samplingWeights[k]=np.log(n_c/(self.n-1+concParam))-(temp[-1]/2)*np.log(2*np.pi*beta_prime)+0.5*np.log(nu_nc/(nu_nc+temp[-1]))+alpha_nc*np.log(beta_nc/beta_prime)+loggamma(alpha_nc+temp[-1]/2)-loggamma(alpha_nc)

                        beta_n=beta+(temp[-1]*nu)/(2*(nu+temp[-1]))*(self.data[j]-mu0)**2
                        samplingWeights[nStar]=np.log(concParam/(self.n-1+concParam))-(temp[-1]/2)*np.log(2*np.pi*beta_n)+alpha*np.log(beta/beta_n)+0.5*np.log(nu/(nu+temp[-1]))+loggamma(alpha+temp[-1]/2)-loggamma(alpha)

                        sampledIndex=catDistLogProb(samplingWeights)
                        if sampledIndex==nStar: #create a new cluster
                            emptyClusterIndex=next((idx for idx, val in np.ndenumerate(N[i]) if val==0)) #Find the first zero cluster
                            #print('add cluster ',emptyClusterIndex)
                            N[i][emptyClusterIndex]+=1
                            S[i][j]=emptyClusterIndex[0]
                            #print('active cluster ', np.unique(C[-1]))


                        else:
                            sampledCluster=activeClusters[sampledIndex]
                            N[i][sampledCluster]+=1
                            S[i][j]=sampledCluster
                    #Sample the concentration param following thanos augmentation method https://users.soe.ucsc.edu/~thanos/notes-2.pdf

                    ##Sample eta, latent var
                    # nStar=len(np.unique(S[i]))
                    # eta=ss.beta.rvs(a=concParam[i]+1,b=self.n)
                    # epsilon=(self.priorParamAlpha[0]+nStar-1)/(self.n*(self.priorParamAlpha[1]-np.log(eta))+self.priorParamAlpha[0]+nStar-1)
                    # if(np.random.random()<epsilon):
                    #     concParam[i]=ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar, scale=1/(self.priorParamAlpha[1]-np.log(eta)))
                    # else:
                    #     concParam[i]=ss.gamma.rvs(a=self.priorParamAlpha[0]+nStar-1, scale=1/(self.priorParamAlpha[1]-np.log(eta)))





                t+=1
                if (temp[-1]==1):
                    break


        return Z

    def SMCAllocFixedConcParamDataTempering(self,numParticles,GibbsStem,ESSthreshold,concParam):
        if self.priorDistLocScale=='Conjugate':
            ESSthreshold=ESSthreshold*numParticles
            n=len(self.data)
            #Initialise with numParticles particles drawn from the prior
            S=[]
            N=[]

            W=-np.log(numParticles)*np.ones(numParticles) #logLlk of the active points
            temp=[]
            temp.append(0)
            #First step : sample from prior
            for i in range(numParticles):
                S.append(np.zeros(self.n,dtype=int))
                N.append(np.zeros(self.n,dtype=int))
                N[-1][0]+=1 #first obs is assigned to first cluster

                for j in range(1,self.n):
                    if np.log(np.random.random())<np.log(concParam)-np.log(concParam+j): #Assign a new cluster :
                        emptyClusterIndex=next((idx for idx, val in np.ndenumerate(N[-1]) if val==0)) #Find the first zero cluster
                        N[-1][emptyClusterIndex]+=1
                        S[-1][j]=emptyClusterIndex[0]
                    else : #assign existing cluster at random among the past allocations
                        newIndex=np.random.choice(self.n,p=N[-1]/j)
                        S[-1][j]=newIndex
                        N[-1][newIndex]+=1
                #W[i]=self.logLikelihoodAllocations(S[i],N[i])
            t=0
            Z=0

            for t in range(n):
                for i in range(numParticles):

                    #Update weights

                    W[i]=(temp[-1]-temp[-2])*self.logLikelihoodAllocations(S[i],N[i]) #log weight update
                Z+=-np.log(numParticles)+logsumexp(W)
                #Normalise weights
                W=W-logsumexp(W)

                #Compute ESS
                ESS=1/(logsumexp(2*W))

                if ESS<ESSthreshold : #Resample
                    for i in range(numParticles):
                        sampIndex=catDistLogProb(W)
                        S[i]=S[sampIndex]
                        N[i]=N[sampIndex]
                    #The weights are then equal
                    W=-np.log(numParticles)*np.ones(numParticles)
