import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize

def Kminimize(fun,k,x0,std,**kwargs):
    # try k random start and pick the best
    bestY = 1e5
    for i in range(k):
        res = minimize(fun,x0+np.random.randn(*x0.shape)*std,**kwargs)
        if res.fun < bestY:
            bestY = res.fun
            bestX = res.x
    return bestX,bestY

reshape_ = lambda x:np.reshape(x,(1,-1))

def EI(x,model,best):
    mu,sigma = model.predict(reshape_(x),return_std=True)
    temp = (mu-best)/sigma
    return -(mu-best)*norm.cdf(temp)-sigma*norm.pdf(temp)

def BO(Fun,ntry,model,initTry,acq,k=10,std=1,globalBest=None,delta=1e-2):
    # Fun is the ground Truth func to max, model is gaussian_process, initTry is a matrix for init Xs
    # acq(x,model,best) is the negative acquisition function, 
    # globalBest is absolute best value used in acquisition function
    # delta is delta added to best for acq
    n0 = initTry.shape[0]
    d = initTry.shape[1]
    y0 = np.zeros(n0)
    for i in range(n0):
        y0[i] = Fun(initTry[i])
    model.fit(initTry,y0)
    j = np.argmax(y0)
    bestX = initTry[j]
    bestY = y0[j] if globalBest is None else globalBest
    X = np.zeros((n0+ntry,d))
    Y = np.zeros(n0+ntry)
    X[:n0] = initTry
    Y[:n0] = y0
    
    for i in range(ntry):
        xNext,_ = Kminimize(acq,k,bestX,std,args=(model,bestY+delta),method='L-BFGS-B')
        yNext = Fun(xNext)
        X[n0+i],Y[n0+i] = xNext,yNext
        model.fit(X[:n0+i+1],Y[:n0+i+1])
        if yNext>bestY:
            bestY = yNext
            bestX = xNext
    
    xLast,yLast = Kminimize(lambda x,model:-model.predict(reshape_(x)),k,bestX,std,args=(model),method='L-BFGS-B')
    return xLast,-yLast
    
''' test
def func2d(x):
    f = -1.0*(np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]) + np.random.randn()/10
    return f

def func2d_noisefree(x):
    f = -1.0*(np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]) 
    return f
    
# best value
func2d_noisefree(np.array([-0.1951, -0.1000]))
model = gp.GaussianProcessRegressor(gp.kernels.Matern(),1e-4)
x,_ = BO(func2d,400,model,np.random.randn(10,2),EI,k=10,std=2,globalBest=None,delta=1e-2)
func2d_noisefree(x)
'''
