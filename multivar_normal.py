
import numpy as np
import scipy as sp


class Normal(object) :
    def __init__( self, mu=np.zeros(1), K=np.eye(1) ) :
        self.mu = mu
        self.K = K
        self.invK = np.linalg.inv( K )
        detK = np.linalg.det(K)
        n = len(mu)
        self.C = np.power( 2 * np.pi * detK, -.5 * n )
        
    def sample( self, sz=None ) :
        return np.random.multivariate_normal( self.mu, self.K, size=sz )
    
    def likelihood(self, x ) :
        dx = x - self.mu
        arg = -0.5 * np.dot( dx, np.dot( self.invK, dx ) )
        return self.C * np.exp( arg )
    
    
    