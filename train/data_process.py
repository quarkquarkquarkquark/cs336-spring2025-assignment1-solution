import os
from tests.adapters import find_max_pairs,get_word_dict,replace_pair,run_train_bpe
from tests.common import FIXTURES_PATH,gpt2_bytes_to_unicode
import sys
from tests.adapters import Tokenizer
import pickle
import regex as re
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path


input_path="./data/TinyStoriesV2-GPT4-valid.txt"
vocab_filepath="./tokenizer_files/ts_train_vocab.pkl"
merges_filepath="./tokenizer_files/ts_train_merges.pkl"
special_tokens=["<|endoftext|>"]



# t=time.time()
# vocab, merges = run_train_bpe(
#     input_path=input_path,
#     vocab_size=10000,
#     special_tokens=["<|endoftext|>"], num_processes=4
# )
# print(time.time()-t)
# with open(vocab_filepath,"wb") as f:
#     pickle.dump(vocab,f)

# with open(merges_filepath,"wb") as f:
#     pickle.dump(merges,f)

# max_len=0
# max_word=""
# for item in vocab.values():
#     if len(item)>max_len:
#         max_len=len(item)
#         max_word=item
# print(max_word)

tokenizer = Tokenizer.from_files(
    vocab_filepath=vocab_filepath,
    merges_filepath=merges_filepath,
    special_tokens=special_tokens
)

with open(input_path) as fin, \
    open("./data/ts_valid_tok.txt","w") as fout:

    for id in tqdm(tokenizer.encode_iterable(fin)):
        fout.write(f"{id} ")


txt_path  = Path('./data/ts_valid_tok.txt') 
bin_path  = Path('./data/ts_valid_tok.bin') 
dtype     = np.uint16            # 目标数据类型
chunk_bytes = 8 * 1024 * 1024     # 8 MiB 一块

carry = b''

with open(txt_path, 'rb') as ftxt, open(bin_path, 'wb') as fbin:
    while True:
        blk = ftxt.read(chunk_bytes)
        if not blk:
            break
        blk = carry + blk
        
        # 找到最后一个空格/换行，把尾巴留给下一块
        last_space = blk.rfind(b' ')
        if last_space == -1:
            last_space = blk.rfind(b'\n')
        if last_space != -1:
            carry = blk[last_space+1:]
            blk = blk[:last_space+1]
        else:
            carry = b''
        
        # 用正则直接提取数字并写入
        nums = np.fromstring(blk, sep=' ', dtype=dtype)
        nums.tofile(fbin)
    
    # 处理文件末尾可能的残余
    if carry:
        nums = np.fromstring(carry, sep=' ', dtype=dtype)
        nums.tofile(fbin)
