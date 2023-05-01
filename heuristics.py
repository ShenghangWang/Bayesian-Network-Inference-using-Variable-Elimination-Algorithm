"""
@Author: Harrison Froedge and Shenghang Wang

Defines utility functions for calculating an
    elimination ordering for variable elimination.
    
Chose not to define a class for these so as to
    make things more generalizable and to keep
    things simple.
    
All orderings executed from 0th element to last element.
"""
import numpy as np
from copy import deepcopy
from collections import Counter
from factors import FactorGroup


def get_all_var_names(factors:FactorGroup, query:str, 
                      include_duplicates:bool = False)->list:
    """
    Wrapper for factors.FactorGroup.getAllVarNames.
    
    param factors: pre-reduced network factors, such that
        observed evidence has been acted upon.

        Rows containing contradictory observation values
        and columns denoting observed variables should be 
        removed.
        
    param query: variable being queried
    
    param include_duplicates: if False, will return the list of
            involved variables without duplicates. By default,
            returns only uniques.
            
    returns all unique variable names present in the 
        factor group
    """
    return [f for f in factors.getAllVarNames(include_duplicates=
                                             include_duplicates) if f != query]
        
    
def random_ordering(factors:FactorGroup, observed, query:str)->np.array:
    """
    Returns a random permutation of the passed variables without
        referencing the FactorGroup.

    param factors: pre-reduced network factors, such that
        observed evidence has been acted upon.

        Rows containing contradictory observation values
        and columns denoting observed variables should be 
        removed.
        
    param observed: list of observed variable names (str)
    
    param query: variable being queried
            
    returns shuffled version of hidden_vars, transformed into
        an np.array
    """
    variables = get_all_var_names(factors, query)
    np.random.shuffle(variables)
    ordering = variables

    return ordering


def fewest_factors_first(factors:FactorGroup, observed, query:str)->np.array:
    """
    Equivalently 'least_outgoing_arcs_first'
    
    Returns variables ordered based on the number of factors which 
    involve them. Variables contained in the fewest factors are placed
    at the beginning of the list. These correspond roughly to those nodes 
    with the least outgoing arcs. Roughly, because observed variables have 
    already been reduced from the factor.
        
    param factors: pre-reduced network factors, such that
        observed evidence has been acted upon.

        Rows containing contradictory observation values
        and columns denoting observed variables should be 
        removed.

    param observed: list of observed variable names (str)
    
    param query: variable being queried
        
    returns ordering such that least frequent variables are evaluated
        first.
    """
    counts = Counter(get_all_var_names(factors, query, include_duplicates=True))
    ordering = [k for k,v in sorted(counts.items(), key=lambda item:item[1])]
    
    return ordering
    
    
def most_factors_first(factors:FactorGroup, observed, query:str)->np.array:
    """
    Equivalently 'most_outgoing_arcs_first'
    
    Returns variables ordered based on the number of factors which 
        involve them. Variables contained in the most factors are placed
        at the beginning of the list. These correspond roughly to those nodes 
        with the most outgoing arcs. Roughly, because observed variables have 
        already been reduced from the factor.
        
    param factors: pre-reduced network factors, such that
        observed evidence has been acted upon.

        Rows containing contradictory observation values
        and columns denoting observed variables should be 
        removed.
        
    param observed: list of observed variable names (str)

    param query: variable being queried
        
    returns ordering such that most frequent variables are evaluated
        first.
    """
    counts = Counter(get_all_var_names(factors, query, include_duplicates=True))
    ordering = [k for k,v in sorted(counts.items(), key=lambda item:item[1],
                                   reverse=True)]
    
    return ordering


def most_incoming_arcs_first(factors:FactorGroup, observed, query:str)->np.array:
    """
    Equivalently: 'largest factors first'
    
    Returns ordering which evaluates largest factors first. These
        correspond roughly to those nodes with the most incoming
        arcs. Roughly, because observed variables have already
        been reduced from the factor.
        
    param factors: pre-reduced network factors, such that
        observed evidence has been acted upon.

        Rows containing contradictory observation values
        and columns denoting observed variables should be 
        removed.
        
    param observed: list of observed variable names (str)

    param query: variable being queried
        
    returns ordering such that those factors with the largest
    conditional probability table are evaluated first.
    """
    counts = {}
    for factor in factors.factors:
        
        if factor.name != query and factor.name not in observed:
            counts[factor.name] = len(factor.cp_table.columns) - 1
        
    ordering = [k for k,v in sorted(counts.items(), key=lambda item:item[1],
                                   reverse=True)]
    
    return ordering


def least_incoming_arcs_first(factors:FactorGroup, observed, query:str)->np.array:
    """
    Equivalently: 'smallest factors first';
    
    Returns ordering which evaluates smallest factors first. These
        correspond roughly to those nodes with the least incoming
        arcs. Roughly, because observed variables have already
        been reduced from the factor.
        
    param factors: pre-reduced network factors, such that
        observed evidence has been acted upon.

        Rows containing contradictory observation values
        and columns denoting observed variables should be 
        removed.
    
    param observed: list of observed variable names (str)
    
    param query: variable being queried
        
    returns ordering such that those factors with the largest
    conditional probability table are evaluated first.
    """
    counts = {}
    for factor in factors.factors:
        
        if factor.name != query and factor.name not in observed:
            counts[factor.name] = len(factor.cp_table.columns) - 1
        
    ordering = [k for k,v in sorted(counts.items(), key=lambda item:item[1])]
    
    return ordering