"""
@Author: Harrison Froedge, Shenghang Wang
"""
from factors import Factor, FactorGroup
from datetime import datetime
import time

class CostTracker():
    
    def __init__(self):
        
        # rough approximation of total computations
        self.computations = 0
        
    
    def trackSums(self, result:Factor)->None:
        """
        Tracks rough estimate of computation count for
            the marginalize factor operation by counting
            the number of probabilities summed to achieve
            a result.
            
            Updates self.computations
        
        param result: result of calling Factor.marginalize()
        
        Returns None
        """
        rows_summed = result.cp_table.shape[0]

        self.computations += rows_summed
        
        return None
        
    
    def trackMerges(self, product:Factor, factors:FactorGroup)->None:
        """
        Tracks rough estimate of computation count for
            the multiply FactorGroup operation by counting
            the number of probabilities multiplied to
            achieve a result.
            
            Updates self.computations
            
        param factors: factors involved in a multiplication step
        param product: result of calling factors.multiply()
        
        Returns None
        """
        num_factors = len(factors.factors)
        size_of_product = product.cp_table.shape[0]
        
        self.computations += num_factors*size_of_product
        
        return None
    
    
    
class LogWriter():
    
    def __init__(self):
        self.start = time.time()
        
        computation_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.string = "Computation started at: {}\n".format(computation_start)
    
    
    def print_elapsed_time(self):
        curr_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.print_message("Computation ends at {}. {} seconds elapsed since log start.".format(curr_time, time.time() - self.start))
    
    
    def end_message(self, delimiter:str="")->None:
        """
        Updates self.string with given delimiter pattern.
        
            Recommended options are '=' and '-'. Prints newline
            if none given
            
        param delimiter: delimiter to write to file to indicate end
            of a given message. Passed delimiter will be repeated n
            times, e.g., '-' produces '----------------------' etc.
        """
        if delimiter:
            border = delimiter*40
            self.string += "\n{}\n".format(border)
        else:
            self.string += "\n"
        
        return None
    
    
    def print_message(self, message:str)->None:
        """
        Updates self.string with given message
        
        param message: string to add to file
        
        returns None
        """
        self.string += "\n{}\n".format(message)
        self.end_message("")
        
        return None
    
    
    def print_factor(self, factor)->None:
        """
        Formats factor to pretty print and then passes 
        the result to self.print_message to add to file.
            
        param factor: factor to add to log
        
        returns None
        """
        self.print_message("{}".format(factor.cp_table))
        
        return None
        
    
    def print_factorGroup(self, factors:FactorGroup)->None:
        """
        Formats factors to pretty print andd then passes the 
            result to self.print_message to add to file.
            
        param factors: a factor group
        
        returns None
        """
        for factor in factors.factors:
            self.print_factor(factor)
            
        return None
        
        
    def write(self)->None:
        """
        Writes contents of self.string to file
        """
        file_name = "log{}.txt".format(time.time())
        
        with open(file_name, "w") as file:
            file.write(self.string)
            
        return None