import re
import numpy as np
from pathlib import Path

# ---------- 参数 ----------
txt_path  = Path('./data/ts_train_tok.txt')       # 原始文本
bin_path  = Path('./data/ts_train_tok.bin')       # 输出二进制
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

