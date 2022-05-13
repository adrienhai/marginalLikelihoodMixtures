# Marginal likelihood estimation for mixture models

### Finite mixtures

The library bayesmixture.py gives all the functions used in order to compute the estimators of the marginal likelihood described in the article. Note that the only type of model currently supported is the normal mixture model with conjugate prior normal-inverse gamma. 
The table below gives a correspondance between the notation used for the hyperparameters in the article, and the one used in the code.

| Algorithm   | Method name                 | Article notation | Tuning parameter |
| ----------- | ---------                   | ---------------- | ---------------- |
| Chib        | chibEstimatorV2             | T          |     numIterGibbs|           
|                         |                           |  burn-in                |     burnIn            |
| Chib Permutation| chibEstimatorV2(permutationChib=True)        |   T          |     numIterGibbs |           
|                         |                           |  burn-in                |     burnIn            |
| Chib Random Permutation | chibEstimatorV2(permutationChib=True)|  T                 |   numIterGibbs  |           
|                          |                                    |  numRandPermut[^2]     | numRandPermut    |
|                         |                           |  burn-in                |     burnIn            |
| Bridge Sampling        | BridgeSampling             |  T0                |     M0             |  
|                         |                           |  T1                |     numIterGibbs             | 
|                         |                           |  T2                |     numIterGibbs[^1]            | 
|                         |                           |  burn-in                |     burnIn            |
| Adaptive SMC        | adaptiveSMC             |      N            |      numParticles            |      
|                         |                           |  M               |     numGibbsStep            | 
| SIS        | SISv2             |      T             |    numSim              |           
| Chib Partition       | ChibPartition             |          T        |        numIterGibbs          | 
|                         |                           |  burn-in                |     burnIn            | 
| Arithmetic Mean        | arithmeticMean             |     T             |     numSim             |           


[^1]:Our implementation of Bridge sampling does not allow for different values of T1 and T2
[^2]:The number of permutations chosen at random to debias Chib's vanilla estimator


### Dirichlet Process Mixtures

The library bayesDPmixture.py gives all the functions used in order to compute the estimators of the marginal likelihood described in the article. Note that the only type of model currently supported is the normal mixture model with conjugate prior normal-inverse gamma. 
The table below gives a correspondance between the notation used for the hyperparameters in the article, and the one used in the code.

| Algorithm   | Method name                 | Article notation | Tuning parameter |
| ----------- | ---------                   | ---------------- | ---------------- |
| Chib        | chibEstimator2v2             | T1          |     numIterGibbs|           
|                         |                           |  T2                |     numIterSIS            |
|                         |                           |  burn-in                |     burnIn            |
| RLR-SIS        | GeyerEstimator(distribution='Newton')             |      T1             |    numIter              |
|                         |                           |  T2                |     numSimPrior            |
|                         |                           |  burn-in                |     burnIn            |
| RLR-Prior        | GeyerEstimator(distribution='prior')             |      T1             |    numIter              |
|                         |                           |  T2                |     numSimPrior            |
|                         |                           |  burn-in                |     burnIn            |
| Harmonic Mean       | harmonicMean             |          T        |        numSim          | 
|                         |                           |  burn-in                |     burnIn            | 
| Arithmetic Mean        | arithmeticMean             |     T             |     numSim             |   
