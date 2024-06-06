from alignment.dtw import *
import numpy as np
import torch
from utils import time_elapsed

# @time_elapsed
def optimized_distance_finder(self_subtraction_matrix,embs,query_embs):
    """
    Credit to Jason :D
    Algorithm: 
        A : [a0, a1, a2, a3, a4 ... ... an]
        B : [b0, b1, b2, b3, b4 ... bm] (n > m)
        Compute distance B and sliding window(A)
        np.sum([a0-b0, a1-b1, a2-b2, a3-b3, a4-b4 ... am-bm]) -> dist0
        np.sum([a1-b0, a2-b1, a3-b2, a4-b3, a5-b4 ... a(m+1)-bm]) -> dist1
        np.sum([a2-b0, a3-b1, a4-b2, a5-b3, a6-b4 ... a(m+2)-bm]) -> dist2
        ...
        a1-b0 is equal to a1-a0 + a0-b0, a2-b1 is equal to a2-a1 + a1-b1, a3-b2 is equal to a3-a2 + a2-b2 ...
        Similarly, 
        a2-b0 is equal to a2-a0 + a0-b0, a3-b1 is equal to a3-a1 + a1-b1, a4-b2 is equal to a4-a2 + a2-b2 ...
    So we only need to compute dist0 and a(0~m) - a(0~m)
    Note: This cannot work on l2 distance because square is incoporated, so we use abs to calcualate distances
    """
    ### Compute dist0
    n = self_subtraction_matrix.size(0)
    m = len(query_embs)
    
    ## dist.shape = m x query_embs.size(-1)
    dist0 = embs[:m] - query_embs ## dont do abs and sum here because we need the signed value to do algorithm
    
    distances = [torch.sum((dist0)**2)]
    for i in range(1, n-m+1):
        ## index diagonallly in self_subtraction_matrix
        distances.append(torch.sum((dist0 + (torch.diagonal(self_subtraction_matrix,i).transpose(0,1)[:m]) )**2))
    min_distance = torch.argmin(torch.stack(distances))
    return min_distance
# @time_elapsed
def align(query_embs,key_embs) -> (int):
    """
    Compute which time window of key_embs is most similar to the query_embs(user's input)
    inputs:
    @ query_embs: Tu , 512
    @ key_embs: Ts , 512
    """
       
    tmp = key_embs.expand(key_embs.size(0),key_embs.size(0),-1)

    self_subtraction_matrix = (tmp - tmp.transpose(0,1))
    """
          a0    a1   a2  a3     a4 ...
    a0  a0-a0 a1-a0 a2-a0 a3-a0 a4-a0
    a1  a0-a1 a1-a1 a2-a1 a3-a1 a4-a1
    a2  a0-a2 a1-a2 a2-a2 a3-a2 a4-a2
    a3  a0-a3 a1-a3 a2-a3 a3-a3 a4-a3
    a4  a0-a4 a1-a4 a2-a4 a3-a4 a4-a4
    ...
    """

    if len(key_embs) < len(query_embs): 
        # print("\033[91m" + f"Typically this shouldn't happen, consider checking the query embs (query embs {name} has shape {query_embs.shape})" + "\033[0m")
        key_embs, query_embs = query_embs, key_embs
    # start_frame = find_min_distance_with_standard(input_embs,query_embs)
    # print(start_frame,opt_start_frame)
    # assert start_frame == opt_start_frame
    # result = input_embs[start_frame:start_frame+len(query_embs)]
    # assert result.shape == query_embs.shape
    opt_start_frame = optimized_distance_finder(self_subtraction_matrix,key_embs,query_embs)
    return opt_start_frame

import unittest

class TestAlign(unittest.TestCase):
    def test_align(self):
        for i in range (100):
            # Create some dummy data for testing
            query_embs = torch.randn(10, 128)
            input_embs = torch.randn(20, 128)

            # Call the align function
            _ = align(query_embs, input_embs)

if __name__ == '__main__':
    torch.manual_seed(42)
    unittest.main()