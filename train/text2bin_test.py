import re
import numpy as np
from pathlib import Path

# ---------- 参数 ----------
txt_path  = Path('./data/ts_valid_tok.txt')       # 原始文本
bin_path  = Path('./data/ts_valid_tok.bin')       # 输出二进制
dtype     = np.uint16            # 目标数据类型
chunk_bytes = 8 * 1024 * 1024   

with open(txt_path,"r") as f:
    s=f.read()
a=np.fromstring(s,sep=' ')

itemsize   = np.dtype(dtype).itemsize
file_bytes = bin_path.stat().st_size
n_elem     = file_bytes // itemsize

data = np.memmap(bin_path, dtype=dtype, mode='r', shape=(n_elem,))

print((data==a).min())