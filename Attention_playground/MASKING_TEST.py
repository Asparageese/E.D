import numpy as np

# Testing the effectiveness of masking single features to single labels with time series consideration.
# Mask such that N layers Considers N features, But as for now. We consider that N layers maps to N heads, Where each head maps to a single node in the input.
# we ask the heads to condsider the relationship between nodes with respect to the masking function. Such if we have 5 nodes head 1 considers N1,N2,N3,N4,N5 
# head 2 considers N1,N2,N3,N4,N05. head 3 considers N1,N2,N3,N04,N05. etc. We say that given some input X of magnitude M. The amount of heads should be M-1.

"""
    [A],[B],[C],[D],[E] , M-1 = 4, N=4

    L1 , ABCDE
    L2 , ABCD0
    L3 , ABC00
    L4 , AB000

MASKING_FORM =  [batch_size, TQ]

"""

