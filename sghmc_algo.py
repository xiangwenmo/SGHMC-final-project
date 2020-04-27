import numpy as np

def data_batch(data, batch_size, seed = 123):
    n = data.shape[0]
    p = data.shape[1]
    if n % batch_size !=0:
        print('%d data dropped during batching' % (n%batch_size))
    sample_size = (n // batch_size)*batch_size
        
    #shuffle
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_batch = n//batch_size
    data = data[idx]
    data = data[:sample_size].reshape(batch_size, p, n_batch)
    return(data, n_batch)

def is_pos_def(A):
    '''function to check if matrix is positive definite'''
    return np.all(np.linalg.eigvals(A) > 0)

def sghmc(gradU, eps, C, Minv, theta_0, V_hat, epochs, burns, data, batch_size, seed = 123):
    
    '''Define SGHMC as dscribed in
    Stochastic Gradient Hamilton Monte Carlo, ICML 2014
    Tianqi Chen, Emily B. Fox, Carlos Guestrin.
    
    n: number of observations in data
    p: dimension of parameters
    
    Inputs:
        gradU: function with parameter(theta, X, y), gradient of U
        
        eps: learning rate
        
        C: friction matrix, with shape (p,p)
        
        Minv: Mass matrix, with shape (p,p)
        
        theta_0: initial value for sampling
        
        V_hat: estimated covariance matrix of stochastic gradient noise
        
        epochs: number of epochs to perform
        
        burns: number of epochs to drop
        
        batch_size: size of a minibatch in an iteration
        
        seed: seed for random generation, default 123
        
    
    Output:
        theta_samp: np.array sampled thetas
    '''
    
    
    np.random.seed(seed)
    
    p = theta_0.shape[0]
    n = data.shape[0]
    
    theta_samp = np.zeros((p, epochs))
    theta_samp[:,0] = theta_0
    
    B_hat = 0.5*eps*V_hat
    
    if not is_pos_def(2*(C-B_hat)*eps):
        print("error: noise term is not positive definite")
        return
    
    sqrt_noise = np.linalg.cholesky(2*(C-B_hat)*eps)
    
    sqrtM = np.linalg.cholesky(np.linalg.inv(Minv))
    r = sqrtM@np.random.normal(size = p).reshape(p, -1)
    
    dat_batch, nbatches = data_batch(data, batch_size)
    for i in range(epochs-1):
        
        theta = theta_samp[:,i]
        r = sqrtM@np.random.normal(size = p).reshape(p, -1)
        
        for batch in range(nbatches):
            theta = theta + (eps*Minv@r).ravel()
            gradU_batch = gradU(theta, dat_batch[:,:, batch], n, batch_size).reshape(p, -1)
            r = r-eps*gradU_batch - eps*C@Minv@r + sqrt_noise@np.random.normal(size = p).reshape(p, -1)
            
        theta_samp[:,i+1] = theta
            
    return theta_samp[:, burns:]