"""
@Author: Harrison Froedge, Shenghang Wang, 
    Joris van Vugt, Moira Berens, Leonieke van den Bulk

Class for the implementation of the variable elimination algorithm.

"""
import pandas as pd
import itertools
import numpy as np
from copy import deepcopy
from factors import Factor, FactorGroup
from reports import LogWriter, CostTracker

class VariableElimination():

    def __init__(self, network):
        """
        Initialize the variable elimination algorithm with the specified network.
        Add more initializations if necessary.

        """
        self.network = network
        

    def run(self, query: str, observed: dict, elim_heuristic, verbose:bool = True, cost_tracker:CostTracker = None)->pd.DataFrame:
        """
        Use the variable elimination algorithm to find out the probability
        distribution of the query variable given the observed variables
        
        Produces a log file and saves it to the working directory.

        Input:
            query:      The query variable
            observed:   A dictionary of the observed variables {'variable': value}
            elim_heuristic: A function that will determine an elimination ordering
                        given the network during the run. Should accept the 
                        initial, pre-reduced network factors and a list of 
                        variables not including the query and observed variables.
            verbose:    if True, prints updates during computation
            cost_tracker: a CostTracker object to track time.
            write_report: if True, writes report to file named 'log<epoch_time>'
                        to the working directory.

        Output: A variable holding the probability distribution
                for the query variable and a log file outputted to the
                current working directory.

        (Added verbose, cost_tracker, and write_report to the param list of this function)
        """
        writer = LogWriter()
        
        variables = set(self.network.nodes)
        writer.print_message("Beginning variable elimination on a network with the following variables: {}".format(variables))  
        
        writer.print_message("Querying on: {}".format(query))
        writer.print_message("Evidence: {}".format(observed))
        
        factors = self.getInitialFactors(variables, observed)
        
        writer.print_message("Reduced factors based on provided evidence. {} variables reduced".format(len(observed)))
        
        writer.print_message("Initial Factors:")
        writer.print_factorGroup(factors)
        
        elim_order = elim_heuristic(factors, observed.keys(), query)
        writer.print_message("Elimination ordering heuristic: {}".format(elim_heuristic))
        writer.print_message("Following elimination ordering: {}".format(elim_order))
    
        writer.print_message("-"*40)

        for X in elim_order:
            
            writer.print_message("Eliminating: {}".format(X))
            
            if verbose:
                print("Eliminating: ", X)
                
            Rs = factors.extractInvolving(X)
            T = Rs.multiply()
            
            writer.print_message("Multiplying following factors which contain X:")
            writer.print_factorGroup(Rs)
            
            writer.print_message("Multiplication produced:")
            writer.print_factor(T)
            
            N = T.marginalize(X)
            
            writer.print_message("Marginalizing above factor on {}".format(X))
            writer.print_factor(N)
                        
            if not N.cp_table.empty:
                # naming convention only important while determining
                # elimination ordering. Thereafter, names only need be unique
                N.name = X
                factors.append(N)
                writer.print_message("Adding marginalized factor to factor list, removing multiplied factors.")
            else:
                writer.print_message("Removing multiplied factors.")
                
            if cost_tracker:
                cost_tracker.trackMerges(T, Rs)
                cost_tracker.trackSums(N)
                writer.print_message("Current total computations: {}".format(cost_tracker.computations))
            
            writer.print_message("Remaining factors:")
            writer.print_factorGroup(factors)
            writer.print_message("-"*40)
        
        writer.print_message("Multiplying remaining factors")
                             
        T = factors.multiply()
        normalized_distribution = T.normalize()
        
        if cost_tracker:
            cost_tracker.trackMerges(T, factors)
            cost_tracker.trackSums(normalized_distribution)
        
        writer.print_message("Normalizing distribution of query variable.")
        writer.print_message("Inferred distribution for {} based on {}:".format(query, observed))
        writer.print_factor(normalized_distribution)
        writer.print_message("Approximate total computations: {}".format(cost_tracker.computations))
        writer.print_elapsed_time()
        writer.write()
        
        return normalized_distribution.cp_table
        
        
    def getInitialFactors(self, variables: set, observed: dict)->FactorGroup:
        """
        Generates and returns the initial dictionary of factors for a
        Bayesian network.
        
        param variables: set of all variables in the network
        param observed: A dictionary of the observed variables {variable: value}
        
        returns dictionary of reduced factors {'variable': pd.DataFrame (factor)}
        """
        factors = FactorGroup(self.network.values)
        
        for var in variables:
            
            new_factor = Factor(var, self.populateFactor(var, observed))

            if len(new_factor.cp_table.columns) > 1:
                # ignores factors which are already fully reduced.
                # note that the key name is somewhat arbitrary,
                # it only matter that it be unique
                factors.append(new_factor)
                
        return factors
        
    
    def populateFactor(self, var: str, observed: dict)->pd.DataFrame:
        """
        Generates factors, implementing reduction at time of generation
        
        param var: node for which to gather probabilities
        param observed: A dictionary of the observed variables {'variable': value}
        
        returns reduced probability table such that rows contradicting
            observations have been removed (pandas.DataFrame)
        """
        df = self.network.probabilities[var]

        for key, val in observed.items():
            # accomodates for non-binary variables
            if key in df:
                # reduction; ignores irrelevant rows, drops observed columns
                # df = df[df["Burglary"] == 'True'].drop(columns=['Burglary'])
                df = df[df[key] == val].drop(columns=[key])

        return df
