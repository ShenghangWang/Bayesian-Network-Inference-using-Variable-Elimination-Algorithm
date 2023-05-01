"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk,
    Harrison Froedge, Shenghang Wang

Entry point for the creation of the variable elimination algorithm in Python 3.
Code to read in Bayesian Networks has been provided. We assume you have installed the pandas package.

"""
from read_bayesnet import BayesNet
from variable_elim import VariableElimination
import heuristics
from reports import CostTracker

if __name__ == '__main__':
    # The class BayesNet represents a Bayesian network from a .bif file in several variables
    net = BayesNet('network_files/earthquake.bif') # Format and other networks can be found on http://www.bnlearn.com/bnrepository/

    # We added several more .bif network files and added them all to the network_files directory.
    
    # These are the variables read from the network that should be used for variable elimination
    print("Nodes:")
    print(net.nodes)
    print("Values:")
    print(net.values)
    print("Parents:")
    print(net.parents)
    print("Probabilities:")
    print(net.probabilities)

    # Make your variable elimination code in the seperate file: 'variable_elim'. 
    # You use this file as follows:
    ve = VariableElimination(net)

    # Set the node to be queried as follows:
    query = 'JohnCalls'

    # The evidence is represented in the following way (can also be empty when there is no evidence): 
    evidence = {}

    # Determine your elimination ordering before you call the run function. The elimination ordering   
    # is either specified by a list or a heuristic function that determines the elimination ordering
    # given the network. Experimentation with different heuristics will earn bonus points. The elimination
    # ordering can for example be set as follows:

    # the following lines were modified from the original file. Additional imports and the @Author tag were also modified
    elim_order = heuristics.fewest_factors_first

    cost_tracker = CostTracker()
    # Call the variable elimination function for the queried node given the evidence and the elimination ordering as follows:   
    distribution = ve.run(query, evidence, elim_order, verbose=True, cost_tracker=cost_tracker)
    
    print(distribution)
    print("Approximate computations: ", cost_tracker.computations)
