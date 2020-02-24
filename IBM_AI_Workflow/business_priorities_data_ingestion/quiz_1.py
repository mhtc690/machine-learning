#! /Users/mahe/anaconda3/bin/python

import numpy as np
from scipy import stats


def get_cubed(lst):
    '''
    INPUT: LIST (containing numeric elements)
    OUTPUT: LIST (cubed value of each even number in originals list)
    return a list containing each element cubed
    '''
    out = []
    for item in lst:
        out.append(item**3)

    return out


def get_squared_evens(lst):
    '''
    INPUT: LIST (containing numeric elements)
    OUTPUT: LIST (squared value of each even number in originals list)
    return squared evens in a list
    '''
    out = []
    for item in lst:
        if item % 2 == 0:
            out.append(item**2)
    return out


def make_dict(lst1, lst2):
    '''
    INPUT: LST1, LST2
    OUTPUT: DICT (LST 1 are the keys and LST2 are the values)
    Given equal length lists create a dictionary where the first
    list is the keys
    '''
    out = {}
    for i in range(len(lst1)):
        out[lst1[i]] = lst2[i]

    return out


def count_characters(string):
    '''
    INPUT: STRING
    OUTPUT: DICT (with counts of each character in input string)

    Return a dictionary of character counts
    '''
    out = {}
    for word in string:
        if word not in out:
            out[word] = 1
        else:
            out[word] += 1

    return out


def calculate_l1_norm(v):
    '''
    INPUT: LIST or ARRAY (containing numeric elements)
    OUTPUT: FLOAT (L1 norm of v)
    calculate and return a norm for a given vector
    '''
    return np.linalg.norm(v, ord=1)


def get_vector_sum(vectorLower, vectorUpper):
    '''
    INPUT: vector lower and upper bounds
    OUTPUT: calculated value for vector sum
    (1) create a vector ranging from 1:150
    (2) transform the vector into a matrix with 10 rows and 15 columns
    (3) print the sum for the 10 rows
    '''
    vector = np.asarray(range(vectorLower, vectorUpper))
    vector = vector.reshape(10, 15)
    return np.sum(vector, axis=1, keepdims=True)


def geometric_distribution(p, k):
    '''
    INPUT: probability of success and trials
    OUTPUT: determined probability
    '''
    return stats.geom.pmf(k, p)


def poisson_distribution(k1, k2):
    '''
    INPUT: probability of event interval
    OUTPUT: determined probability
    '''
    p_lt_7 = 0
    for i in range(k2+1):
        p_lt_7 += stats.poisson.pmf(k1, i)
    return (1-p_lt_7)


def gaussian_distribution(loc_val, scale_val, cdf_val):
    '''
    INPUT: loc, scale, and cdf values
           loc: mean, scale: standard deviation
           cdf: Cumulative distribution function
    OUTPUT: determined probability
    '''
    return 1 - stats.norm.cdf(cdf_val, loc_val, scale_val)


def matrix_multiplication(A, B):
    '''
    INPUT: LIST (of length n) OF LIST (of length n) OF INTEGERS
    LIST (of length n) OF LIST (of length n) OF INTEGERS
    OUTPUT: LIST OF LIST OF INTEGERS
    (storing the product of a matrix multiplication operation)
    Return the matrix which is the product of matrix A and matrix B
    where A and B will be (a) integer valued (b) square matrices
    (c) of size n-by-n (d) encoded as lists of lists, e.g.
    A = [[2, 3, 4], [6, 4, 2], [-1, 2, 0]] corresponds to the matrix
    | 2 3 4 |
    | 6 4 2 |
    |-1 2 0 |
    You may not use numpy. Write your solution in straight python
    '''
    n = len(A)
    out = [[0] * n for i in range(n)]
    for row in range(n):
        for colume in range(n):
            for i in range(n):
                out[row][colume] += A[row][i] * B[i][colume]

    return out


A = [[2, 3, 4], [6, 4, 2], [-1, 2, 0]]
print(np.dot(A, A).tolist())
print(matrix_multiplication(A, A))
