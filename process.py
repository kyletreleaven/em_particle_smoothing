

class MarkovChain(object) :
    """ interface """
    @classmethod
    def sample_trans( cls, state ) :
        raise Exception('not implemented')
    
    @classmethod
    def likelihood_trans( cls, next_state, state ) :
        raise Exception('not implemented')
    
    
class HiddenMarkovModel(MarkovChain) :
    """ (additional interface) """
    def sample_output( self, state ) :
        raise Exception('not implemented')
    
    def likelihood_output(self, y, state ) :
        raise Exception('not implemented')
    
    
class Parameter(object) :
    """
    an interface for the SPSA algorithm
    """
    def perturbation(self) :
        raise Exception('not implemented')
    
    def perturbation_twosided( self ) :
        raise Exception('not implemented')
    
    def update(self, param1, param2, diff_likelihood, gamma ) :
        raise Exception('not implemented')
    
    
    
    
class Parametrized(object) :
    def set_params(self, params ) :
        raise Exception('not implemented')
    
    def get_params(self ) :
        raise Exception('not implemented')




class CompoundHMM(HiddenMarkovModel) :
    pass

class CompoundParametrized(Parametrized) :
    pass






