
import numpy as np

from process import *
from multivar_normal import *


class AircraftState(object) :
    def __init__(self, pos=None, vel=None ) :
        if pos is None : pos = np.zeros(3)
        if vel is None : vel = np.zeros(3)
        self.pos = pos
        self.vel = vel
        
    def copy(self) :
        res = type( self )()
        res.pos = self.pos.copy()
        res.vel = self.vel.copy()
        return res

class AircraftTimedModel(object) :
    @classmethod
    def update( cls, state, dt, inplace=False ) :
        if inplace :
            state_next = state
        else :
            state_next = state.copy()
            
        state_next.pos += dt * state.vel
        return state_next
    
    @classmethod
    def disturb(cls, state, w, dt, inplace=False ) :
        if inplace :
            state_next = state
        else :
            state_next = state.copy()
            
        state_next.pos += 0.5 * np.power( dt, 2. ) * w
        state_next.vel += dt * w
        return state_next
    

class Aircraft(AircraftState) :
    def __init__(self, pos=None, vel=None ) :
        if pos is None : pos = np.zeros(3)
        self.pos = pos
        
        if vel is None : vel = np.zeros(3)
        self.vel = vel
        
    def update(self, dt ) :
        AircraftTimedModel.update( self, dt, inplace=True )
        return self
    
    def disturb(self, w, dt ) :
        AircraftTimedModel.disturb( self, w, dt, inplace=True )
        return self








if __name__ == '__main__' :
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
#
    import matplotlib.patches as patches
    import mpl_toolkits.mplot3d.art3d as art3d
    
    from multivar_normal import *
    
    
    NOISEMEAN   = np.zeros(3)
    NOISECOVAR  = np.diag([1.,1.,1.])
    #NOISECOVAR  = np.diag([0.,0.,0.])
    
    N = 1000
    dt = .5
    X = np.zeros((3,N))
    #
    veh = Aircraft()
    noise = Normal( NOISEMEAN, NOISECOVAR )
    
    #x = aircraft_state()
    veh.vel[0] = 100.
    for k in range(N) :
        X[:,k] = veh.pos
        w = noise.sample()
        #w = np.zeros(3)
        veh.update( dt ).disturb( w, dt )
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
#
    #rad = np.linalg.norm([x,y])
    #circ = patches.Circle( (0,0), rad, linestyle='dashed', fill=False )
    #ax.add_patch( circ )
    #art3d.pathpatch_2d_to_3d( circ, z )
#    
    trials = X      # avoid renaming stuff
    #ax.scatter( trials[0,:], trials[1,:], trials[2,:] )
    ax.plot( trials[0,:], trials[1,:], trials[2,:] )
    #ax.set_xlim3d(-rad,rad)
    #ax.set_ylim3d(-rad,rad)
    plt.show()
    
    
    
    
    
    