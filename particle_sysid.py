
import itertools

from process import *

import bootstrap_filter as bootstrap


class WrapParamModel(object) :
    def __init__(self, HMM, params ) :
        self.HMM = HMM
        self.params = params
        
    def sample_trans(self, state ) :
        return self.HMM.sample_trans( state, self.params )
    
    def sample_output(self, state ) :
        return self.HMM.sample_output( state, self.params )
    
    def likelihood_output(self, y, state ) :
        return self.HMM.likelihood_output( y, state, self.params )


def particle_sysid_filter( n, y, samples, pHMM, params, delta, gamma, k=None ) :
    a, b = params.perturbation_twosided( delta )
    
    HMM1 = WrapParamModel( pHMM, a )
    HMM2 = WrapParamModel( pHMM, b )
    
    _, alpha1 = bootstrap.bootstrap_evolve( samples, y, HMM1, k )
    _, alpha2 = bootstrap.bootstrap_evolve( samples, y, HMM2, k )
    #print alpha1, alpha2
    
    J1 = np.log( np.average( alpha1 ) )
    J2 = np.log( np.average( alpha2 ) )
    
    params_next = params.copy().update( a, b, J2 - J1, gamma )
    
    # then actually filter ya know...
    HMM_next = WrapParamModel( pHMM, params_next )
    samples_next = bootstrap.bootstrap_filter( n, y, samples, HMM_next, k )
    
    return samples_next, params_next
    
    
""" step sizes generator """
def step_sizes( n ) :
    delta = np.power( n, -1./5 )
    gamma = np.power( n, -3./4 )
    return delta, gamma

def step_gen() :
    for i in itertools.count( 1 ) :
        yield step_sizes( i )
        
        
        
        
if __name__ == '__main__' :
    
    from aircraft_lingauss import *
    from radar import *
    
    from multivar_normal import *
    
    
    
    
    class SystemState(object) :
        def __init__(self, pos=None, vel=None, d_theta=None ) :
            self.pos = pos
            self.vel = vel
            self.d_theta = d_theta
            
        def copy(self) :
            res = type( self )( self.pos.copy(), self.vel.copy(), self.d_theta )
            return res
        
        
    class SystemModel(object) :
        TIMESTEP = .1
        
        def __init__( self ) :
            self.bumpy = Normal( np.zeros(3), np.diag( np.ones(3) ) )
        
        def sample_trans(self, state, params ) :
            cstate = AircraftState( state.pos, state.vel )
            rstate = RadarState( state.d_theta )
            
            cstate_next = AircraftTimedModel.update( cstate, self.TIMESTEP )
            w = self.bumpy.sample()
            AircraftTimedModel.disturb( cstate_next, w, self.TIMESTEP, inplace=True ) 
                                        
            rstate_next = RadarTimedModel.update( rstate, self.TIMESTEP, params )
            w = RadarTimedModel.sample_disturbance( params )
            RadarTimedModel.disturb( rstate_next, w, self.TIMESTEP, inplace=True )
            
            return SystemState( cstate_next.pos, cstate_next.vel, rstate_next.d_theta )
        
        def sample_output(self, state, params ) :
            rstate = RadarState( state.d_theta )
            radar_noise = RadarTimedModel.sample_radar_noise( rstate, params )
            output = RadarTimedModel.noisy_radar( state.pos, radar_noise )
            return output
        
        def likelihood_output(self, output, state, params ) :
            rstate = RadarState( state.d_theta )
            return RadarTimedModel.likelihood( output, state.pos, rstate, params )
    
    
    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as patches
    import mpl_toolkits.mplot3d.art3d as art3d
    
    
    SYS = SystemModel()
    #
    xy_dist = 10000.
    bearing = 2 * np.pi * np.random.rand()
    altitude = 10000.
    pos = np.array([ xy_dist * np.cos(bearing), xy_dist * np.sin(bearing), altitude ])
    #
    speed = 100.
    heading = 2 * np.pi * np.random.rand()
    vel = np.array([ speed * np.cos(heading), speed * np.sin(heading), 0. ])
    cstate = AircraftState( pos, vel )
    
    rstate = RadarState()
    state = SystemState( cstate.pos, cstate.vel, rstate.d_theta )
    #
    params = RadarParams()
    params_est = RadarParams( .9, 1. )       # see what happens
    
    class data(object) :
        def __repr__(self) :
            res = 'truth=%s, ' % repr( self.state )
            res += repr( self.samples )
            return res
        
    xx = data()
    xx.params = params
    xx.state = state
    xx.params_est = params_est
    xx.samples = [ state ]
    
    
    def step( x, delta, gamma ) :
        params = x.params
        state = x.state
        #
        params_est = x.params_est
        samples = x.samples
        
        # true actions
        state_next = SYS.sample_trans( state, params )
        y = SYS.sample_output( state_next, params )
        
        # filtering
        samples_next, params_next = particle_sysid_filter( 50, y, samples, SYS, params_est, delta, gamma, k=3 )
        
        x_next = data()
        x_next.params = params
        x_next.state = state_next
        x_next.y = y
        #
        x_next.params_est = params_next
        x_next.samples = samples_next
        return x_next
    
    """ just visualization below """
    def scatter_points( cloud, truth=None, ax=None, marker=None ) :
        #plt.close('all')
        if ax is None :
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        if truth is not None :
            x,y,z = truth
            
            ax.scatter( x, y, z, marker='x' )
            if False :
                rad = np.linalg.norm([x,y])
                circ = patches.Circle( (0,0), rad, linestyle='dashed', fill=False )
                ax.add_patch( circ )
                art3d.pathpatch_2d_to_3d( circ, z )
                ax.plot( [ 0, x ], [ 0, y ], [ 0, z ] )
                ax.set_xlim3d(-rad,rad)
                ax.set_ylim3d(-rad,rad)
            
        cloud = np.vstack( cloud ).transpose()
        ax.scatter( cloud[0,:], cloud[1,:], cloud[2,:], marker=marker )
        return ax
    
    def display( x, ax=None ) :
        pos = x.state.pos
        X = [ state.pos for state in x.samples ]
        ax = scatter_points( X, pos, ax, marker='o' )
        plt.show()
        return ax
    
    
    
    if True :
        """ run the simulation """
        POS = []
        
        def plot_traj( poses, ax=None ) :
            if ax is None :
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

            tableau = np.vstack( POS )
            ax.plot( tableau[:,0], tableau[:,1], tableau[:,2] )
            return ax
        
        steps = step_gen()
        def next() :
            global xx
            global delta, gamma
            global model, evolve, alpha
            
            delta, gamma = steps.next()
            xx = step( xx, delta, gamma )
            
            model = WrapParamModel( SYS, xx.params_est )
            evolve, alpha = bootstrap.bootstrap_evolve( xx.samples, xx.y, model, k=3 )
            
        for i in range(10000) :
            next()
            POS.append( xx.state.pos )
            
            if i % 50 == 0 :
                some_measurements = [ SYS.sample_output( xx.state, xx.params ) for i in range(100) ]
                some_measurements = [ RadarTimedModel.recover_xyz( *m ) for m in some_measurements ]
                
                plt.close('all')
                ax = plot_traj( POS )
                scatter_points( some_measurements, xx.state.pos, ax, marker='^' )
                display( xx, ax )
                
                plt.show()
            
            #raw_input()
            
            #print alpha
            print xx.params_est
            print 'delta=%f, gamma=%f' % ( delta, gamma )
            
        #display( xx )
            
        
        
    
    
    
