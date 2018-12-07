import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

data = np.loadtxt('datos_observacionales.dat')

def dxdt(sigma, x, y, z):
    return sigma * (y-x)
def dydt(rho, x, y, z):
    return x * (rho - z) - y
def dzdt(beta, x, y, z):
    return x * y - beta * z
def model_odf(w,t,params):
    dx = dxdt(params[0],w[0],w[1],w[2])
    dy = dydt(params[1],w[0],w[1],w[2])
    dz = dzdt(params[2],w[0],w[1],w[2])
    dw = [dx,dy,dz]
    return dw
def model(t, params, w_in=data[0,1:]):
    w_solve = odeint(model_odf,w_in,t, args=(params,))
    x = w_solve[:,0]
    y = w_solve[:,1]
    z = w_solve[:,2]
    return x,y,z
def loglikelihood(t, f, param, sigma=1.0):
    d = f.T -  model(t, param)
    d = d/sigma
    d = -0.5 * np.sum(d**2)
    return d
def logprior(param):
    return np.sum(np.where(param>30,-np.inf,0))
def H(t,f,param,momentum):
    K = np.sum(momentum)** 2.0 / 2.0
    V = - loglikelihood(t,f,param) - logprior(param)
    return K + V
def der_loglikelihood(t,f,param):
    n = len(param)
    res = []
    d = 0.001
    for i in range(n):
        l_1 = loglikelihood(t,f,param+d)
        l_2 = loglikelihood(t,f,param-d)
        res.append((l_1+l_2)/2*d)
    res = np.array(res)
    return res
def leapfrog(t,f,param,momentum):
    dt = 0.0001
    param_n = np.array(param)
    momentum_n = np.array(momentum)
    for i in range(3):
        momentum_n = momentum_n + der_loglikelihood(t,f,param) * dt/2.0
        param_n = param_n + momentum * dt
        momentum_n = momentum_n + der_loglikelihood(t,f,param) * dt/2.0
    momentum_n = - momentum_n
    return param_n, momentum_n
def MC(d):
    param = [np.random.random(3)]
    momentum = [np.random.normal(loc=7, size=3)]
    t = d[:,0][1:]
    w = d[:,1:4][1:]
    for i in range(1,1000):
        p , m = leapfrog(t,w,param[i-1], momentum[i-1])
        E_new = H(t,w,p,m)
        E_old = H(t,w,param[i-1],momentum[i-1])
        
        alpha = np.random.random()
        r = np.exp(-(E_new-E_old))
        #print(p,m)
        if (r>alpha):
            param.append(p)
        else:
            param.append(param[i-1])
        momentum.append(np.random.normal(loc=10,size=3))
    return param

p = np.array(MC(data))

plt.figure(figsize=(11,4))

plt.subplot(1,3,1)
_ = plt.hist(p[:,0])
m_s = np.mean(p[:,0])
s = np.std(p[:,0])
plt.title('sigma\n'+'mean='+str(m_s)+'\n'+'std='+str(s))

plt.subplot(1,3,2)
_ = plt.hist(p[:,1])
m_r = np.mean(p[:,1])
s = np.std(p[:,1])
plt.title('rho\n'+'mean='+str(m_r)+'\n'+'std='+str(s))

plt.subplot(1,3,3)
_ = plt.hist(p[:,2])
m_b = np.mean(p[:,2])
s = np.std(p[:,2])
plt.title('beta\n'+'mean='+str(m_b)+'\n'+'std='+str(s))

plt.savefig('histogramas.png')

