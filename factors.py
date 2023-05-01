"""
@Author: Harrison Froedge, Shenghang Wang
"""

from __future__ import annotations
import pandas as pd
import itertools
import numpy as np
from copy import deepcopy
from warnings import warn


class FactorGroup():
    
    def __init__(self, values: dict, *factors:Factor):
        """
        Defines a group of factors and relevant operations
            which can be applied on that group.
        
        param values: dict of possible values the variables
            initially included in this FactorGroup can take on.
            {'variable':[possible values]}. Is not updated as
            variables are removed.
            
        param factors: factors to include in the group
        """
        self.values = values
        self.factors = []
        self.append(factors)
        
        # defined by approximate row/column operations in
        # multiply and marginalize        
        self.computations = 0
        
    
    def append(self, *factors:Factor)->None:
        """
        Adds an aribtrary number of factors to the
            end of the factors list.
        """
        if factors[0]:
            
            for factor in factors:

                if not isinstance(factor, Factor):

                        raise ValueError("Can only append items of type Factor" +
                                         " to FactorGroup, not " +
                                         "{}".format(type(factor)))

                self.factors.append(factor)
        
        return None
    
    
    def extractInvolving(self, var: str)->FactorGroup:
        """
        Collects all factors which contain var in their
            conditional probability table and returns
            them as a new, separate FactorGroup.
            Calling will update self.factors
            
        param var: variable (column name) to search on
            
        returns new FactorGroup with factors involving variable
        """
        Rs = FactorGroup(self.values)
        updated_factors = []
        
        for factor in self.factors:
            
            if var in factor.cp_table:
                Rs.append(factor)
            else:
                updated_factors.append(factor)
                
        self.factors = updated_factors
        
        return Rs
    
    
    def getAllVarNames(self, include_duplicates:bool=False)->list:
        """
        Retrieves and returns all variables
            involved in this FactorGroup, i.e., all
            variables contained as a column in a
            cp_table of at least one Factor in
            self.factors

        param include_duplicates: if False, will return the list of
            involved variables without duplicates. By default,
            returns only uniques.

        returns set of variables involved in the FactorGroup's
            cp_tables (without duplicates if get_uniques=True)
        """
        variables = [item for sublist in [
            list(f.cp_table.drop(columns=["prob"]).columns) 
            for f in self.factors] for item in sublist]

        if not include_duplicates:
            variables = list(set(variables))
        
        return variables
        
    
    def getProductTable(self, Rs: list)->pd.DataFrame:
        """
        Generates table which results from multiplying given factors.
            Initializes probabilities to NaN
            
        param Rs: list of factors
        
        returns conditional probability table containing all unique
            variables found in factors. If v many unique variables exist
            across factors, and each variable can take on n many values,
            then the resulting table will have n^v many rows, with 
            values in the prob column left undeclared.
        """
        new_column_names = self.getAllVarNames()
        
        vals = {k: self.values[k] for k in new_column_names}

        table = pd.DataFrame(columns=new_column_names, data=list(itertools.product(*vals.values())))

        return table
    
    
    def multiply(self)->Factor:
        """
        Multiplies all factors by matching the probabilities
            from their corresponding rows. If v many unique
            variables exist among factors, and each variable
            can take on n many values, multiply produces a 
            single factor with n^v many rows and v columns.
            
            Generates conditional probability table with
            getProductTable, and then defines the probability
            column by matching 
        
        param Rs: factors to be multiplied; assumes some
            variable(s) is common between factors.
        
        returns a new factor: the product of factors
        """        
        product = Factor("", self.getProductTable(self.factors))
        
        # JOINs factors with product table
        
        # Extends each factor to have n^v rows, where each row in product
        # with variable values matching a row in the factor are duplicated
        # (but only with the columns existing in that factor).
        # This means that each factor will have the same number of rows as
        # product, and their row indices exactly correspond to product's.
        # These extensions are stored in merges
        merges = [product.merge(factor) for factor in self.factors]
        
        # multiplies probabilities of resulting merges together row-by-row 
        # across every extended factor in merge.
        prob_products = []
        for index, row in product.iterrows():
            prob_products.append(np.product([cp_table['prob'].iloc[index] for cp_table in merges]))
        
        product.cp_table['prob'] = prob_products
        
        return product
        
        
class Factor():
    
    def __init__(self, name: str, cp_table: pd.DataFrame):
        """
        Defines a factor, used to represent nodes in a 
            Bayesian network and to perform variable
            elimination.
        
        param name: a name with which to reference
            the factor. Can be the names given in network.nodes.
        param cpt: Conditional Probability Table for a
            node in the network
        """
        self.name = name
        self.cp_table = cp_table
    
    
    def normalize(self)->Factor:
        """
        Normalizes the factor's conditional probability
            table such that it's total probability sums
            to one.
        """
        if len(self.cp_table.columns) > 2:
            warn("Normalizing factor with conditional probabilities" +
                " may not be mathematically consistent")
            
        prob = self.cp_table['prob']
        normalized_prob = prob.divide(prob.sum())  

        normalized_cp_table = deepcopy(self.cp_table)            
        normalized_cp_table['prob'] = normalized_prob

        self.cp_table = normalized_cp_table
        
        return self
    
    
    def marginalize(self, variable:str)->Factor:
        """
        The elimination step of variable elimination. Sums out
        a variable from a given factor by summing the probabilities
        of rows which are identical except for the value of the
        passed variable to make a single row. The summed rows,
        along with the column of the subject variable, are discarded.

        param variable: a variable in the factor to be eliminated.

        Results in a new cp_table with the subject variable's column
            summed out and with reduced rows. For an initial
            cp_table of size v x n, the resulting cp_table will be 
            of size (v-1) x (n/2).
        """
        if variable not in self.cp_table:
            raise ValueError("Something went wrong." +
                             " Variable to be eliminated not found in factor")
            
        if len(self.cp_table.columns) <= 2:
            
            # summing over a factor with only one variable will
            # completely eliminate it, so return empty df to
            # indicate that nothing should now be added to
            # the factor dictionary
            
            summed_f = pd.DataFrame()
            
        else:
            # get list of columns minus variable and prob column
            groupby = list(set(self.cp_table.columns).difference(set([variable,'prob'])))
            reindex = groupby + ['prob']
            agg_funcs = {c:'first' for c in groupby}
            agg_funcs['prob'] = 'sum'

            # will 'merge' rows together and sum their probabilities
            # if every value in each column specified by groupby is
            # equivalent
            summed_f = self.cp_table.groupby(groupby).aggregate(
                agg_funcs).reset_index(level=[i for i in range(
                len(groupby))], drop=True)

        self.cp_table = summed_f
        
        return self
    
    
    def merge(self, factor: Factor)->Factor:
        """
        Applies a merge with factor without affecting the
            conditional probability table of either factor.
        
        param factor: factor to merge with
        """
        return self.cp_table.merge(factor.cp_table, how='left')
    
    
    def iterrows(self)->pd.DataFrame.iterrows:
        """
        Returns an iterrows generator for the factor's
            conditional probability table
        """
        return self.cp_table.iterrows()