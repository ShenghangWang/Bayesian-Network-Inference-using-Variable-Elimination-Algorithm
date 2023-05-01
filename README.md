# Bayesian-Network-Inference-using-Variable-Elimination-Algorithm

The Variable Elimination algorithm is an important technique in the field of probabilistic graphical models that allows for the efficient computation of the probability distribution of a query variable given evidence. This technique has applications in various fields including artificial intelligence, machine learning, and data analysis.

This project implements the Variable Elimination algorithm in Python, with the goal of allowing users to easily apply this technique to their own Bayesian networks. The code is designed to take a Bayesian network as input and use it to compute the probability distribution of a query variable given evidence.

The implementation includes a class called VariableElimination, which contains methods for initializing the algorithm with a network, generating initial factors, populating factors, and running the algorithm with an elimination heuristic. The algorithm uses a factor graph to represent the joint probability distribution of a set of variables and performs variable elimination to remove variables that are not relevant to the query. The algorithm also includes the ability to use an elimination heuristic to determine the order in which variables should be eliminated, which can significantly improve performance.

The code also includes functionality for logging and tracking the cost of computations, which can be useful for analyzing the performance of the algorithm and optimizing it for a particular use case.
