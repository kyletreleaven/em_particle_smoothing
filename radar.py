
import random, math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d

from process import *
from multivar_normal import *



"""
range/altitude/bearing:

#radar measures
slant range (sr):     actual radar-to-object distance
altitude (alt):       measurement of y_true
bearing (theta):      polar version of ( x_true, y_true )
"""

class RadarParams(Parameter) :
    # process
    BEARING_DRIFT_ALPHA = .75
    BEARING_DRIFT_SIGMA = .2
    
    # observation
    SIGMA_SR = 50.                   # slant range measurement noise
    SIGMA_ALT = 250.                    # altitude measurement noise
    #
    SIGMA_THETA = 5. * np.pi / 180.     # bearing measurement noise (after bias)
    
    def __init__(self, drift_alpha=None, drift_sigma=None, sigma_sr=None, sigma_alt=None, sigma_theta=None ) :
        if drift_alpha is None : drift_alpha = self.BEARING_DRIFT_ALPHA
        if drift_sigma is None : drift_sigma = self.BEARING_DRIFT_SIGMA
        if sigma_sr is None : sigma_sr = self.SIGMA_SR
        if sigma_alt is None : sigma_alt = self.SIGMA_ALT
        if sigma_theta is None : sigma_theta = self.SIGMA_THETA
        
        self.drift_alpha = drift_alpha
        self.drift_sigma = drift_sigma
        self.sigma_sr = sigma_sr
        self.sigma_alt = sigma_alt
        self.sigma_theta = sigma_theta
        
    def __repr__(self) :
        return '(a=%f,s=%f)' % ( self.drift_alpha, self.drift_sigma )
        
    def copy( self ) :
        res = type(self)( self.drift_alpha, self.drift_sigma, self.sigma_sr, self.sigma_alt, self.sigma_theta )
        return res
        
    def perturbation_twosided(self, delta ) :
        u1 = random.choice([-1, 1])
        u2 = random.choice([-1, 1])
        
        a = self.copy()
        a.drift_alpha -= delta * u1
        a.drift_sigma = np.exp( np.log( self.drift_sigma ) - delta * u2 )
        
        b = self.copy()
        b.drift_alpha += delta * u1
        b.drift_sigma = np.exp( np.log( self.drift_sigma ) + delta * u2 )
        
        return a, b
    
    def update(self, param1, param2, diff, gamma, inplace=False ) :
        DESCENT_DIR = 1.0       # likelihood MAXIMIZATION
        
        if inplace :
            param_next = self
        else :
            param_next = self.copy()
            
        alpha1, sigma1 = param1.drift_alpha, param1.drift_sigma
        alpha2, sigma2 = param2.drift_alpha, param2.drift_sigma
        
        param_next.drift_alpha += DESCENT_DIR * gamma * diff / ( alpha2 - alpha1 )
        #
        arg_sigma = np.log( self.drift_sigma ) + DESCENT_DIR * gamma * diff / ( np.log( sigma2 ) - np.log( sigma1 ) )
        
        #print sigma1, np.log( sigma1 ), sigma2, np.log( sigma2 ), diff
        #print arg_sigma
        if math.isnan( arg_sigma ) : raise Exception('stop yo!')
        param_next.drift_sigma = np.exp( arg_sigma )
        #param_next.drift_sigma += gamma * diff / ( sigma2 - sigma1 )
        return param_next




class RadarState(object) :
    def __init__(self, d_theta=None ) :
        if d_theta is None : d_theta = 0.
        self.d_theta = d_theta
        
    def copy(self) :
        res = type( self )()
        res.d_theta = self.d_theta
        return res


class RadarTimedModel(object) :
    @classmethod
    def update(cls, state, dt, params, inplace=False ) :
        if inplace :
            state_next = state
        else :
            state_next = state.copy()
            
        state_next.d_theta *= params.drift_alpha
        return state_next
    
    @classmethod
    def sample_disturbance(cls, params ) :
        return params.drift_sigma * np.random.normal()
        
    @classmethod
    def disturb(cls, state, w, dt, inplace=False ) :
        if inplace :
            state_next = state
        else :
            state_next = state.copy()
            
        state_next.d_theta += dt * w
        return state_next
        
        
    """ output interface """
    @classmethod
    def sample_radar_noise(cls, state, params ) :
        d_sr = params.sigma_sr * np.random.normal()
        d_alt = params.sigma_alt * np.random.normal()
        d_theta = state.d_theta + params.sigma_theta * np.random.normal()
        return d_sr, d_alt, d_theta
    
    @classmethod
    def noisy_radar(cls, pos, radar_noise ) :
        sr, alt, theta = cls.perfect_radar( pos )
        ( d_sr, d_alt, d_theta ) = radar_noise
        return sr + d_sr, alt + d_alt, theta + d_theta
    
    @classmethod
    def likelihood(cls, radar_output, pos_actual, radar_state, params ) :
        sr_o, alt_o, theta_o = radar_output
        sr_t, alt_t, theta_t = cls.perfect_radar( pos_actual )
        #
        radar_noise = ( sr_o - sr_t, alt_o - alt_t, theta_o - theta_t )
        radar_noise = np.array( radar_noise )
        
        mu = np.array([ 0., 0., radar_state.d_theta ])
        K = np.diag([ params.sigma_sr ** 2, params.sigma_alt ** 2, params.sigma_theta ** 2 ])
        #K = np.dot( K, K )  # will this really fix it?
        return Normal( mu, K ).likelihood( radar_noise )
        
        
    
    """ convenience function """
    @classmethod
    def perfect_radar( cls, pos ) :
        sr      = np.linalg.norm( pos )
        alt     = pos[2]
        theta   = np.arctan2( pos[1], pos[0] )
        return sr, alt, theta
    
    @classmethod
    def recover_xyz(cls, sr, alt, theta ) :
        try :
            r = np.sqrt( np.power( sr, 2. ) - np.power( alt, 2. ) )
        except :
            print sr, alt, theta
            
        x = r * np.cos( theta )
        y = r * np.sin( theta )
        return np.array([ x, y, alt ])





class DynamicRadar(RadarState) :
    def __init__(self, d_theta=None, params=None ) :
        if d_theta is None : d_theta = 0.
        if params is None : params = RadarParams()
        
        self.d_theta = d_theta
        self.params = params
        
    def copy(self) :
        res = type( self )( self.d_theta, self.params )
        return res
    
    def set_params(self, params ) :
        self.params = params
        return self
    
    def get_params(self ) :
        return self.params
        
        
    """ simulation interface """
    def update(self, dt ) :
        RadarTimedModel.update( self, dt, self.params, inplace=True )
        return self
    
    def noisy_update(self, dt ) :
        self.update( dt )
        w = RadarTimedModel.sample_disturbance( self.params )
        RadarTimedModel.disturb( self, w, dt, inplace=True )
        return self
    
    def noisy_radar( self, pos ) :
        radar_noise = RadarTimedModel.sample_radar_noise( self, self.params )
        res = RadarTimedModel.noisy_radar( pos, radar_noise )
        return res
    
    """ convenience functions """
    perfect_radar = RadarTimedModel.perfect_radar
    recover_xyz = RadarTimedModel.recover_xyz




if __name__ == '__main__' :

    """ setup  """
    dt = 1.0
    theta = 2 * np.pi * np.random.rand()
    unit = np.array([ np.cos(theta), np.sin(theta) ])
    x, y = 5000. * unit
    z = 10000. + 1000. * np.random.normal()
    #x, y = 1000. * np.random.normal(size=2)
    pos = np.array([x,y,z])
    
    radar_config = RadarParams()
    radar = DynamicRadar( None, radar_config )
    for k in range(100) : radar.noisy_update( dt )
    print 'starting bearing offset (truth): %f' % radar.d_theta
    
    
    """ trajectory simulation """
    N = 100
    trials = [] #np.zeros( ( 3, 100 ) )
    for k in range(N) :
        reading = radar.noisy_radar( pos )
        trials.append( radar.recover_xyz( *reading ) )
        radar.noisy_update( dt )
        
    
    """ just visualization below """
    def scatter_points( cloud, truth=None, ax=None ) :
        if ax is None :
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        if truth is not None :
            x,y,z = truth
            
            rad = np.linalg.norm([x,y])
            circ = patches.Circle( (0,0), rad, linestyle='dashed', fill=False )
            ax.add_patch( circ )
            art3d.pathpatch_2d_to_3d( circ, z )
            ax.plot( [ 0, x ], [ 0, y ], [ 0, z ] )
            
            ax.set_xlim3d(-rad,rad)
            ax.set_ylim3d(-rad,rad)
            
        cloud = np.vstack( cloud ).transpose()
        ax.scatter( cloud[0,:], cloud[1,:], cloud[2,:] )
        return ax
    
    ax = scatter_points( trials, pos )
    plt.show()
    
    def tuplify( np_array ) :
        return tuple([ x for x in np_array ])
    
    if False :
        print 'position:\t(%f,%f,%f)' % tuplify( pos )
        sr, alt, theta = DynamicRadar.perfect_radar( pos )
        print 'SR,ALT,THETA:\t(%f,%f,%f)' % (sr,alt,theta)
        sr_hat, alt_hat, theta_hat = radar.noisy_radar( pos )
        print 'RADAR READING:\t(%f,%f,%f)' % (sr_hat,alt_hat,theta_hat)
        pos_hat = DynamicRadar.recover_xyz( sr_hat, alt_hat, theta_hat )
        print 'recovered:\t(%f,%f,%f)' % tuplify( pos_hat )
    
    
    
    
    
    

