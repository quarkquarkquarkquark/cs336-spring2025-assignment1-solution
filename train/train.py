import os
import numpy as np
from pathlib import Path
from tests.data_preprocess import process_1d_array
from tqdm import tqdm
import torch

# model, optimizer
from tests.layers import transformer_lm, cross_entropy, adamw, get_lr_cosine_schedule, grad_clip
from .config import train_config
from torch.utils.tensorboard import SummaryWriter
from tests.utils import save_checkpoint, load_checkpoint

from datetime import datetime
formatted = datetime.now().strftime('%y%m%d_%H%M%S')

config=train_config()
log_dir=os.path.join(config.log_dir,formatted)
print(log_dir)
writer = SummaryWriter(log_dir=log_dir)  
ds_path = Path(config.input_path)       # 输出二进制
eval_path = Path(config.eval_path)
dtype = np.uint16            # 目标数据类型

itemsize = np.dtype(dtype).itemsize
file_bytes = ds_path.stat().st_size
n_elem = file_bytes // itemsize

dataset = np.memmap(ds_path, dtype=dtype, mode='r', shape=(n_elem,))

itemsize = np.dtype(dtype).itemsize
file_bytes = eval_path.stat().st_size
n_elem = file_bytes // itemsize
eval_ds = np.memmap(eval_path, dtype=dtype, mode='r', shape=(n_elem,))

model=transformer_lm(
    vocab_size = config.vocab_size,
    context_length = config.context_length,
    d_model = config.d_model,
    num_layers = config.num_layers,
    num_heads = config.num_heads,
    d_ff = config.d_ff,
    rope_theta = config.rope_theta,
    device=config.device
).to(config.device)

print("Transformer paras: ",sum([p.numel() for p in model.parameters()]))
print("Dataset toks :", len(dataset))

optimizer=adamw(model.parameters(),lr=config.lr , betas= config.betas)

def evaluation():
    with torch.no_grad():
        eval_loss=0
        eval_step=100
        for i in tqdm(range(eval_step),leave=False):
            inputs_ids, targets = process_1d_array(eval_ds, config.batch_size, config.context_length, device=config.device)
            pred = model(inputs_ids.to(torch.int32))
            loss = cross_entropy()(pred.view(config.batch_size*config.context_length,config.vocab_size),targets.to(torch.int64).flatten())
            eval_loss+=loss.item()
        return eval_loss/eval_step



for step in tqdm(range(config.training_steps)):
    inputs_ids, targets = process_1d_array(dataset, config.batch_size, config.context_length, device=config.device)
    new_lr=get_lr_cosine_schedule(step, max_learning_rate=config.lr , min_learning_rate = config.lr/100,
            warmup_iters = config.warmup_iters, cosine_cycle_iters = config. cosine_cycle_iters)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    # sample
    # inputs_ids, targets = process_1d_array(dataset, config.batch_size, config.context_length, device=config.device)
    
    pred = model(inputs_ids.to(torch.int32))
    
    loss = cross_entropy()(pred.view(config.batch_size*config.context_length,config.vocab_size),targets.to(torch.int64).flatten())
    
    writer.add_scalar("loss/train", loss.item(), step)
    writer.add_scalar("lr", new_lr, step)
    
    optimizer.zero_grad()
    loss.backward()
    grad_norm=grad_clip(model.parameters(),1)
    writer.add_scalar("grad_norm", grad_norm.item(), step)
    optimizer.step()
    if (step+1)%config.eval_steps ==0:
        eval_loss=evaluation()
        writer.add_scalar("evaluation_loss", eval_loss, step)
        
        

out=os.path.join(config.output_dir,f"ckpt_{formatted}.pt")
save_checkpoint(model,optimizer,config.training_steps,out)
    