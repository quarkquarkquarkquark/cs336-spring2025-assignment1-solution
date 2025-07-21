# import time
# from tests.adapters import find_max_pairs
# import numpy as np
# from finder import find_max_pairs_cython

# pairs_cnt={}
# vocab={}
# for i in range(5000000):
#     pairs_cnt[(i,i)]=i
#     vocab[i]=str(i)
# t=time.time()
# for i in range(10):
#     find_max_pairs(pairs_cnt,vocab)
# print(time.time()-t)
# keys = np.array(list(pairs_cnt.keys()))
# values = np.array(list(pairs_cnt.values()))
# t=time.time()
# for i in range(10):
#     find_max_pairs_cython(pairs_cnt,vocab)
# print(time.time()-t)