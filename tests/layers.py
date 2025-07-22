import torch
from torch import nn,Tensor
from einops import einsum,rearrange
from jaxtyping import Float, Int, Bool
import numpy as np
from collections.abc import Callable, Iterable
from typing import Optional

def init_paras(layer):
    pass

class c_linear(nn.Module):
    def __init__(
            self,
            in_features : int,
            out_features : int,
            device : torch.device | None=None,
            dtype : torch.dtype| None=None,
    ):
        super().__init__()
        self.d_in=in_features
        self.d_out=out_features
        self.W=nn.Parameter(torch.randn(out_features,in_features,device=device,dtype=dtype))

    def forward(self,x : Float[Tensor,"... d_in"]):
        y=einsum(x,self.W,"... d_in, d_out d_in -> ... d_out")
        return y

class embedding(nn.Module):
    def __init__(self,
                 num_embeddings : int, 
                 embedding_dim : int, 
                 device : torch.device| None=None, dtype=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.embedding_mat=nn.Parameter(torch.randn(num_embeddings,embedding_dim,dtype=dtype,device=device))

    def forward(self,token_ids : Int[Tensor, "..."]):
        return self.embedding_mat[token_ids]

class rmsnorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps=eps
        self.d_model=d_model
        self.weight=nn.Parameter(torch.randn(d_model,device=device,dtype=dtype))
    
    def forward(self,x: Float[Tensor, "... d_model"]):
        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms=torch.sqrt(torch.sum(x**2,axis=-1,keepdim=True)/self.d_model+self.eps)
        rms=rms.to(in_dtype)
        y=einsum(x/rms,self.weight,"... d_model, d_model -> ... d_model")
        return y
    

class swishglu(nn.Module):
    def __init__(self,
                 d_model : int,
                 d_ff : int,
                 device : torch.device | None=None,
                 dtype : torch.dtype | None=None,
                 ):
        super().__init__()
        self.w1=nn.Parameter(torch.randn(d_ff,d_model,dtype=dtype,device=device))
        self.w2=nn.Parameter(torch.randn(d_model,d_ff,dtype=dtype,device=device))
        self.w3=nn.Parameter(torch.randn(d_ff,d_model,dtype=dtype,device=device))

    def forward(self,x: Float[Tensor, "... d_model"]):
        x1=einsum(x,self.w1,"... d_model, d_ff d_model -> ... d_ff")
        x1=x1*torch.sigmoid(x1)
        x3=einsum(x,self.w3,"... d_model, d_ff d_model-> ... d_ff")
        y=einsum((x1*x3),self.w2,"... d_ff, d_model d_ff-> ... d_model")
        return y

class rope(nn.Module):
    def __init__(self,
                 theta:float,
                 d_k:int, 
                 max_seq_len:int, 
                 device: torch.device|None=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        assert d_k%2==0
        self.max_seq_len=max_seq_len
        theta_list=1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.t=torch.arange(max_seq_len).unsqueeze(1)*theta_list
        self.cos=torch.cos(self.t)
        self.sin=torch.sin(self.t)
        
    
    def forward(self, x : Float[Tensor, "... d_k"], token_positions : Float[Tensor, "..."]):
        assert x.shape[-2]<=self.max_seq_len
        x=rearrange(x,"... (a b) -> ... a b", a=self.d_k//2,b=2)
        even=x[...,0]
        odd=x[...,1]
        cos=self.cos[token_positions]
        sin=self.sin[token_positions]
        # print("!!!!",cos.shape)
        out=torch.concat([(cos*even-sin*odd).unsqueeze(-1),(sin*even+cos*odd).unsqueeze(-1)],axis=-1)
        out=rearrange(out,"... a b -> ... (a b)")
        return out
        
class softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x : Float[Tensor,"..."],i:int):
        c=torch.max(x,dim=i,keepdim=True).values
        x-=c
        return torch.exp(x)/torch.sum(torch.exp(x),dim=i,keepdim=True)
    
class attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                Q: Float[Tensor,"... seq_len d"],
                K : Float[Tensor, "... seq_len d"],
                V: Float[Tensor,"... seq_len d_v"],
                mask: Float[Tensor, "seq_len seq_len"]
                ):
        d=K.shape[-1]
        dtype=Q.dtype
        qk=einsum(Q,K,"... q_seq_len d,... k_seq_len d -> ... q_seq_len k_seq_len")/np.sqrt(d)
        qk+=torch.where(mask,0,-torch.inf)
        qkv=einsum(softmax()(qk,-1),V,"... a b, ... b d_v -> ... a d_v")
        return qkv

class casual_mha(nn.Module):
    def __init__(self,
                 d_model: int,n_head : int,
                 device: torch.device| None=None,
                 dtype: torch.dtype| None=None,
                 ):
        super().__init__()
        self.n_head=n_head
        self.d_model=d_model
        self.d_head=d_model//n_head
        self.q_proj=nn.Parameter(torch.randn(d_model,d_model,device=device,dtype=dtype))
        self.k_proj=nn.Parameter(torch.randn(d_model,d_model,device=device,dtype=dtype))
        self.v_proj=nn.Parameter(torch.randn(d_model,d_model,device=device,dtype=dtype))
        self.output_proj=nn.Parameter(torch.randn(d_model,d_model,device=device,dtype=dtype))

    
    def forward(self,
                Q : Float[Tensor, "... seq_len d_model"],
                K : Float[Tensor, "... seq_len d_model"],
                V : Float[Tensor, "... seq_len d_model"], 
                casual: bool,
                mask : Float[Tensor,"... seq_len seq_len"]| None=None,
                position_layer=None,
                token_positions=None,
                ):
        seq_len=Q.shape[-2]
        q=einsum(self.q_proj,Q,"... a b, ... sl b -> ... sl a")
        k=einsum(self.k_proj,K,"... a b, ... sl b -> ... sl a")
        v=einsum(self.v_proj,V,"... a b, ... sl b -> ... sl a")
        q=rearrange(q,"... sl (h d_h) -> ... sl h d_h",h=self.n_head,d_h=self.d_head)
        k=rearrange(k,"... sl (h d_h) -> ... sl h d_h",h=self.n_head,d_h=self.d_head)
        v=rearrange(v,"... sl (h d_h) -> ... sl h d_h",h=self.n_head,d_h=self.d_head)
        if position_layer:            
            q=position_layer(q.transpose(1,2),token_positions=token_positions).transpose(1,2)
            k=position_layer(k.transpose(1,2),token_positions=token_positions).transpose(1,2)

        if mask is not None:
            mask=torch.where(mask,0,-torch.inf)
        else:
            if casual:
                mask=torch.where(torch.tril(torch.ones(seq_len,seq_len)).to(dtype=torch.bool),0,-torch.inf)
        qk=einsum(q,k,"... seq_len_q n_head d_head, ... seq_len_k n_head d_head -> ... n_head seq_len_q seq_len_k")/np.sqrt(self.d_head)
        qk+=mask
        qk=softmax()(qk,-1)
        qkv=einsum(qk,v,"... n_head seq_len_q seq_len_k, ... seq_len_k n_head d_head -> ... seq_len_q n_head d_head")
        qkv=rearrange(qkv,"... n_head d_v -> ... (n_head d_v)")
        return einsum(self.output_proj,qkv,"... a b, ... b -> ... a")


class transformer_block(nn.Module):
    def __init__(self,
                 d_model : int,
                 num_heads : int,
                 d_ff : int,
                 max_seq_len : int,
                 device : torch.device| None=None,
                 dtype : torch.dtype | None=None,
                 theta : float = 10000.0,
                 ):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.max_seq_len=max_seq_len
        self.attn=casual_mha(d_model,num_heads,device=device,dtype=dtype)
        self.pos_emb_layer=rope(theta=theta,d_k=d_model//num_heads,max_seq_len=max_seq_len)
        self.ln1=rmsnorm(d_model,device=device,dtype=dtype)
        self.ffn=swishglu(d_model,d_ff,device=device,dtype=dtype)
        self.ln2=rmsnorm(d_model,device=device,dtype=dtype)

    def forward(self,x):
        token_positions=torch.arange(x.shape[1])
        x_norm=self.ln1(x)
        x=x+self.attn(x_norm,x_norm,x_norm,position_layer=self.pos_emb_layer,token_positions=token_positions,casual=True)
        x=x+self.ffn(self.ln2(x))
        return x


# vocab_size: int,
# context_length: int,
# d_model: int,
# num_layers: int,
# num_heads: int,
# d_ff: int,
# rope_theta: float,

class transformer_lm(nn.Module):
    def __init__(self,vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta):
        super().__init__()
        self.token_embeddings=embedding(vocab_size,d_model)
        self.layers=nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta
            ))
        self.ln_final=rmsnorm(d_model=d_model)
        self.lm_head=c_linear(d_model,vocab_size)
        
    def forward(self,in_indices):
        x=self.token_embeddings(in_indices)
        for layer in self.layers:
            x=layer(x)
        x=self.ln_final(x)
        x=self.lm_head(x)
        # x=softmax()(x,-1)
        return x
        
class cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "batch_size vocab_size"], target : Int[Tensor,"batch_size"]):
        c=torch.max(x,dim=-1,keepdim=True).values
        x-=c
        log_den=torch.log(torch.sum(torch.exp(x),dim=-1,keepdim=True))
        log_prob=log_den-x
        log_prob_tgt=torch.gather(log_prob,index=target.unsqueeze(1),dim=-1).squeeze(1)
        return log_prob_tgt.mean()
    

class adamw(torch.optim.Optimizer):
    def __init__(self,weights,lr,betas=(0.9,0.95),weight_decay=0.01,eps=1e-8):
        defaults={"lr" : lr}
        super().__init__(weights,defaults)
        self.beta_1,self.beta_2=betas
        for group in self.param_groups:
            group['m']=[torch.zeros_like(p) for p in group['params']]
            group['v']=[torch.zeros_like(p) for p in group['params']]

        self.lamb=weight_decay
        self.eps=eps

    def step(self,closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr=group['lr']
            for idx,p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state=self.state[p]
                t=state.get("t",1)
                grad=p.grad.data
                group['m'][idx]=self.beta_1*group['m'][idx]+(1-self.beta_1)*grad
                group['v'][idx]=self.beta_2*group['v'][idx]+(1-self.beta_2)*(grad**2)
                lr_t=lr*np.sqrt(1-self.beta_2**t)/(1-self.beta_1**t)
                p.data-=lr_t*group['m'][idx]/(torch.sqrt(group['v'][idx])+self.eps)
                p.data-=lr*self.lamb*p.data
                state["t"]=t+1

def get_lr_cosine_schedule(
            it: int,
            max_learning_rate: float,
            min_learning_rate: float,
            warmup_iters: int,
            cosine_cycle_iters: int,
):
    if it<warmup_iters:
        return it/warmup_iters*max_learning_rate
    if it<=cosine_cycle_iters:
        theta=(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*np.pi
        return min_learning_rate+(np.cos(theta)+1)/2*(max_learning_rate-min_learning_rate)
    return min_learning_rate

def grad_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    grad_norm=0
    for p in parameters:
        if p.grad!=None: 
            grad_norm+=(p.grad**2).sum()
    grad_norm=torch.sqrt(grad_norm)
    if grad_norm>max_l2_norm:
        for p in parameters:
            if p.grad!=None: 
                p.grad=p.grad*max_l2_norm/(grad_norm+eps)
