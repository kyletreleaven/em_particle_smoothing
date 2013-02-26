
"""
TODO: There seems to be a bias in the filter, and I'm not sure where it's coming from;
potentially from calculating alpha incorrectly?
"""

import itertools

import numpy as np
import networkx as nx

from process import *


def bootstrap_evolve( samples, y, HMM, k=None ) :
    if k is None :
        upsample = samples
    else :
        # a better way to do this would be with chairman assignment
        #times = np.random.multinomial( N, weights )
        #iters = [ itertools.repeat( s, t ) for s, t in zip( samples, times ) ]
        upsample = [ s for s in itertools.chain( *itertools.repeat( samples, k ) ) ]
        
    evolve = [ HMM.sample_trans( s ) for s in upsample ]
    # the Gorden paper (bootstrap filter) suggests only weighting by likelihood...
    alpha = [ HMM.likelihood_output( y, s ) for s in evolve ]
    
    return evolve, alpha


def bootstrap_resample( n, samples, weights ) :
    times = np.random.multinomial( n, weights )
    iters = [ itertools.repeat( s, t ) for s, t in zip( samples, times ) ]
    res = [ s for s in itertools.chain( *iters ) ]
    return res      #, [ 1./n for k in range(n) ]


def bootstrap_filter( n, y, samples, HMM, k=None ) :
    """
    sample N new samples from sample_dict, multi-nomial according to weights
    resample down to n, using importance weights according to observation y;
    process modeled by ( chain, sensor )
    """
    # still seems to have a funky bias
    evolve, alpha = bootstrap_evolve( samples, y, HMM, k )
    
    # normalize alpha into multi-nomial weights
    total_weight = sum( alpha )
    for i, a in enumerate( alpha ) :
        alpha[i] = a / total_weight
        
    res = bootstrap_resample( n, evolve, alpha )
    return res
    





if __name__ == '__main__' :
    
    import matplotlib.pyplot as plt
    
    
    chain_graph = nx.DiGraph()
    numstates = 5
    for i in range(numstates) :
        alpha = np.random.exponential( size=numstates )
        alpha = alpha / np.sum( alpha )
        for j, c in enumerate( alpha ) :
            chain_graph.add_edge( i, j, weight=c )
            
    sensor_graph = nx.DiGraph()
    numchar = 3
    for i in range(numstates) :
        alpha = np.random.exponential( size=numchar )
        alpha = alpha / np.sum( alpha )
        for j, c in enumerate( alpha ) :
            sensor_graph.add_edge( i, 'output %d' % j, weight=c )
            
            
    class Machine(object) :
        def __init__(self, digraph ) :
            self.digraph = digraph
        
        def sample( self, state ) :
            edges = self.digraph.edges( state, data=True )
            pvals = [ data.get( 'weight', 0. ) for _,__,data in edges ]
            select = np.random.multinomial( 1, pvals )
            j = sum([ i*k for i,k in enumerate( select ) ])
            #print select, j
            _,j,__ = edges[j]
            return j
        
        def likelihood( self, j, i ) :
            return self.digraph[i][j]['weight']
        
        
    import multivar_normal as gauss
    class WienerProcess(MarkovChain) :
        def __init__(self, mu, K ) :
            self.noise = gauss.Normal( mu, K )
            
        def sample_trans( self, state ) :
            return state + self.noise.sample()
        
        def likelihood_trans(self, next_state, state ) :
            return self.noise.likelihood( next_state - state )
            
            
    class WienerHMM(HiddenMarkovModel) :
        def __init__(self, mu_trans, K_trans, K_noise ) :
            self.proc = WienerProcess( mu_trans, K_trans )
            
            mu = np.zeros( len( mu_trans ) )
            self.noise = gauss.Normal( mu, K_noise )
            
        def sample_trans(self, state ) :
            return self.proc.sample_trans( state )
        def likelihood_trans(self, next_state, state ) :
            return self.proc.likelihood_trans( next_state, state )
        
        def sample_output(self, state ) :
            return state + self.noise.sample()
        def likelihood_output(self, y, state ) :
            return self.noise.likelihood( y - state )
        
        
        
        
    if True :
        dim = 2
        HMM = WienerHMM( np.zeros(dim), np.eye(dim), np.eye(dim) )
        #chain = WienerProcess( np.zeros(dim), np.eye(dim) )
        #sensor = WienerProcess( np.zeros(dim), np.eye(dim) )
        x0 = np.zeros(dim)
        
    else :
        chain = Machine( chain_graph )
        sensor = Machine( sensor_graph )
        x0 = 0



    class data(object) :
        def __repr__(self) :
            res = 'truth=%s, ' % repr( self.state )
            res += repr( self.samples )
            return res
        
    xx = data()
    xx.state = x0
    xx.samples = [ x0 ]
    #xx.weights = [ 1. ]
    
    
    def step( x ) :
        state = x.state
        samples = x.samples
        
        state_next = HMM.sample_trans( state )
        y = HMM.sample_output( state_next )
        
        samples_next = bootstrap_filter( 50, y, samples, HMM, k=3 )
        
        x_next = data()
        x_next.state = state_next
        x_next.y = y
        x_next.samples = samples_next
        return x_next
    
    def display( x ) :
        plt.close('all')
        state = x.state
        plt.scatter( state[0], state[1], marker='x' )
        
        sample_states = [ s for s in x.samples ]
        X = [ s[0] for s in sample_states ]
        Y = [ s[1] for s in sample_states ]
        plt.scatter( X, Y )
        
        plt.show()
        
        
    for i in range(1000) :
        xx = step( xx )
    display( xx )
        
        
        
        
        
        