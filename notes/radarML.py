



if __name__ == '__main__' :
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    def normpdf( x, mu, var ) :
        c = 2. * np.pi * var
        c = np.power( c, -1./2 )
        arg = -0.5 * np.power( x - mu, 2. ) / var
        return c * np.exp( arg )
    
    def radar_loglikelihood_constrained( z, z_o, R_o, varz, varR ) :
        term1 = np.power( np.abs( z ) - R_o, 2. ) / varR
        term2 = np.power( z - z_o, 2. ) / varz
        return term1 + term2
        
    
    varR = .1
    varz = 1.
    
    
    z_o = -1
    R_o = 10.
    Z = np.linspace(-12,12,1000)
    f = radar_loglikelihood_constrained( Z, z_o, R_o, varz, varR )
    
    plt.close('all')
    plt.plot( Z, f )
    
    
    
    