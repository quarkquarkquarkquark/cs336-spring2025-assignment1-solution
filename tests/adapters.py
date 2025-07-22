from __future__ import annotations
import pickle

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable, Iterator
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor
import regex as re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
import threading
from tqdm import tqdm
import time
import numpy as np
from finder import find_max_pairs_cython,update_pairs_cnt,encode_apply_merges
from .layers import c_linear,embedding,rmsnorm,swishglu,rope,softmax,attention,casual_mha
from .layers import transformer_block,transformer_lm
from .layers import cross_entropy,adamw,get_lr_cosine_schedule,grad_clip


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    layer=c_linear(d_in,d_out)
    layer.W.data=weights
    y=layer(in_features)
    return y


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    emb_layer=embedding(vocab_size,d_model)
    emb_layer.embedding_mat.data=weights
    return emb_layer(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    print(w1_weight.shape,w2_weight.shape,w3_weight.shape)
    swi_layer=swishglu(d_model,d_ff)
    swi_layer.w1.data=w1_weight
    swi_layer.w2.data=w2_weight
    swi_layer.w3.data=w3_weight
    return swi_layer(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return attention()(Q,K,V,mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha=casual_mha(d_model=d_model,n_head=num_heads)
    mha.q_proj.data=q_proj_weight
    mha.k_proj.data=k_proj_weight
    mha.v_proj.data=v_proj_weight
    mha.output_proj.data=o_proj_weight
    return mha(in_features,in_features,in_features,casual=True)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    rope_layer=rope(theta=theta,d_k=d_model//num_heads,max_seq_len=max_seq_len)
    mha=casual_mha(d_model=d_model,n_head=num_heads)
    mha.q_proj.data=q_proj_weight
    mha.k_proj.data=k_proj_weight
    mha.v_proj.data=v_proj_weight
    mha.output_proj.data=o_proj_weight
    return mha(in_features,in_features,in_features,casual=True,position_layer=rope_layer,token_positions=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope_layer=rope(theta,d_k,max_seq_len)
    return rope_layer(in_query_or_key,token_positions)

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    tf_block=transformer_block(d_model,num_heads,d_ff,max_seq_len,theta=theta)
    w_d={}
    for k,v in weights.items():
        if "ln" not in k:
            w_d[k.replace(".weight","")]=v
        else:
            w_d[k]=v
    tf_block.load_state_dict(w_d)
    return tf_block(in_features)

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    tf_lm=transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )
    # for n,p in weights.items():
    #     print("~",n,p.shape)
    # for n,p in tf_lm.state_dict().items():
    #     print("!",n)
    w_d={}
    for n,p in weights.items():
        if n=="ln_final.weight":
            w_d["ln_final.weight"]=p
        elif n=="lm_head.weight":
            w_d["lm_head.W"]=p
        elif n=="token_embeddings.weight":
            w_d["token_embeddings.embedding_mat"]=p
        elif "ln" in n:
            w_d[n]=p
        else:
            w_d[n.replace(".weight","")]=p
    
    tf_lm.load_state_dict(w_d)
    return tf_lm(in_indices)

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms_layer=rmsnorm(d_model,eps)
    rms_layer.weight.data=weights
    return rms_layer(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    
    return softmax()(in_features,dim)


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    cr=cross_entropy()
    return cr(inputs,targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grad_clip(parameters,max_l2_norm)
    return


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return adamw


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    return get_lr_cosine_schedule(it,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


class Tokenizer():
    def __init__(self,
        vocab:dict[int,bytes],
        merges:list[tuple[bytes,bytes]],
        special_tokens:list[str]|None=None
    ):

        self.vocab=vocab
        self.reverse_vocab={value:key for key,value in vocab.items()}
        self.merges=merges
        self.merges_dict={(self.reverse_vocab[item[0]],self.reverse_vocab[item[1]]):self.reverse_vocab[item[0]+item[1]] for item in self.merges}
        if special_tokens == None:
            special_tokens=[]

        special_tokens=sorted(special_tokens, key=len, reverse=True)
        self.special_tokens=[item.encode('utf-8') for item in special_tokens]
        self.special_tokens_len=[len(tk) for tk in self.special_tokens]
        self.cache={}
        self.t=0
        self.merge_cnt=0
        

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary 
        and list of merges(in the same format that your BPE training code output) and 
        (optionally) a list of specialtokens. This method should accept the following 
        additional parameters:"""
        with open(vocab_filepath,"rb") as f:
            vocab=pickle.load(f)

        with open(merges_filepath,"rb") as f:
            merges=pickle.load(f)

        return cls(vocab=vocab,merges=merges,special_tokens=special_tokens)
    
    def encode_pre_token(self, text:str)->list[int]:
        text=text.encode('utf-8')
        str_len=len(text)
        res=[]
        i=0
        while i<str_len:
            f=0
            for idx,tok in enumerate(self.special_tokens):
                sp_len=self.special_tokens_len[idx]
                if text[i:i+sp_len]==tok:
                    i+=sp_len
                    try:
                        res.append(self.reverse_vocab[tok])
                    except:
                        res.append(b"<|unknown|>")
                    f=1
                    break
            if f:
                continue
            else:
                res.append(self.reverse_vocab[int.to_bytes(text[i])])
                i+=1
        t=time.time()

        res=encode_apply_merges(self.merges_dict,res)
        self.merge_cnt+=1
        self.t+=time.time()-t
        return res
        
    def encode(self, text:str)->list[int]:
        """Encode an input text into a sequence of token IDs."""
        
        s_pat="|".join([re.escape(item.decode('utf-8')) for item in self.special_tokens])
        chunk_ind=[0]
        sp_list=set([])
        if len(self.special_tokens)>0:
            for match in re.finditer(s_pat,text):
                chunk_ind.append(match.span()[0])
                chunk_ind.append(match.span()[1])
                sp_list|=set([match.span()])
        chunk_ind.append(len(text))
        # print(chunk_ind)
        res=[]
        for i in range(len(chunk_ind)-1):
            st=chunk_ind[i]
            ed=chunk_ind[i+1]
            s=text[st:ed]
            if (st,ed) in sp_list:
                res+=self.encode_pre_token(s)
            else:
                pat=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                w_cnt=0
                for item in re.finditer(pat,s):
                    w_cnt+=1
                    if item.group() not in self.cache:
                        self.cache[item.group()]=self.encode_pre_token(item.group())
                    
                    res+=self.cache[item.group()]
        return res


    def encode_iterable(self, iterable: Iterable[str])->Iterator[int]:
        """Given an iterable ofstrings (e.g., a Python file handle), return a generator 
        that lazily yields token IDs. This isrequired for memory-eﬀicient tokenization of 
        large files that we cannot directly load intomemory."""
        for i in iterable:
            output=self.encode(i)
            for idx in output:
                yield idx

    def decode(self, ids:list[int])->str:
        """Decode a sequence of token IDs into text."""
        s=b"".join([self.vocab[i] for i in ids])
        try:
            return s.decode("utf-8")
        except:
            return s

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab,merges,special_tokens)

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize(chunk_list):
    pat=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_cnt={}
    for chunk in chunk_list:
        iterlist=re.finditer(pat,chunk)
        for match in iterlist:
            s=match.group().encode("utf-8")
            if s in word_cnt:
                word_cnt[s]+=1
            else:
                word_cnt[s]=1

    return word_cnt

def get_word_dict(input_path,special_tokens,num_processes=4):
    with open(input_path,"rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        s_pat="|".join([re.escape(item) for item in special_tokens])

        with ProcessPoolExecutor(max_workers=num_processes) as exe:
            futures = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_list=re.split(s_pat,chunk)
                future = exe.submit(pre_tokenize, chunk_list)
                futures.append(future)

            rc=Counter()
            
            for future in futures:
                rc.update(future.result())

    return rc

# def find_max_pairs(pairs_cnt,vocab):
#     max_cnt=-1
#     max_pair=(-1,-1)
#     max_pair_str=None
#     for item in pairs_cnt.keys():
#         if pairs_cnt[item]>max_cnt or (pairs_cnt[item]==max_cnt and (vocab[item[0]],vocab[item[1]])>max_pair_str):
#             max_cnt=pairs_cnt[item]
#             max_pair=item
#             max_pair_str=(vocab[item[0]],vocab[item[1]])
#     return max_pair,max_cnt

def find_max_pairs(pairs_cnt,vocab):
    keys = np.array(list(pairs_cnt.keys()))
    values = np.array(list(pairs_cnt.values()))
    
    max_val = values.max()
    max_mask = (values == max_val)
    
    max_keys = keys[max_mask]
    str_tuples = [(vocab[k[1]], vocab[k[0]]) for k in max_keys]
    lex_last = np.lexsort(np.array(str_tuples).T)[-1]  # 选字典序最大的
    
    return tuple(max_keys[lex_last]), max_val

def replace_pair(lst, pair, new_val):
    a, b = pair
    res=[]
    i=0
    while i<len(lst):
        if i!=len(lst)-1 and lst[i]==pair[0] and lst[i+1]==pair[1]:
            res.append(new_val)
            i+=2
            continue

        res.append(lst[i])
        i+=1
    return res

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # init
    if "num_processes" in kwargs:
        num_processes=kwargs['num_processes']
    else:
        num_processes=4
    print("num_processes: ",num_processes)
    word_cnt=get_word_dict(input_path,special_tokens=special_tokens,num_processes=num_processes)
    merges=[]
    word_b_pairs={}

    pairs_cnt={}
    pairs_n=0

    cnt=256
    vocab={i:bytes([i]) for i in range(256)}
    for idx,st in enumerate(special_tokens):
        vocab[cnt+idx]=st.encode('utf-8')
    cnt+=len(special_tokens)
    print("word_cnt: ",len(word_cnt))
    t=time.time()
    for k in word_cnt:
        word_b_pairs[k]=[k[i] for i in range(len(k))]
        for i in range(len(word_b_pairs[k])-1):
            if (k[i],k[i+1]) in pairs_cnt:
                pairs_cnt[(k[i],k[i+1])]+=word_cnt[k]
            else:
                pairs_cnt[(k[i],k[i+1])]=word_cnt[k]
    print("word cnt time:",time.time()-t)
    pbar=tqdm(total=vocab_size,initial=len(vocab)-1)

    get_max_time=0
    word_cnt=dict(word_cnt)
    while len(vocab)<vocab_size:
        t=time.time()
        merge_pair,max_cnt=find_max_pairs_cython(pairs_cnt,vocab)
        get_max_time+=time.time()-t
        vocab[cnt]=vocab[merge_pair[0]]+vocab[merge_pair[1]]
        merges.append((vocab[merge_pair[0]],vocab[merge_pair[1]]))
        cnt+=1
        word_b_pairs,pairs_cnt=update_pairs_cnt(pairs_cnt,vocab,word_cnt,word_b_pairs,cnt,merge_pair)        
        # for k in word_cnt:
        #     if vocab[cnt-1] in k:
        #         for i in range(len(word_b_pairs[k])-1):
        #             pairs_cnt[(word_b_pairs[k][i],word_b_pairs[k][i+1])]-=word_cnt[k]
        #             if pairs_cnt[(word_b_pairs[k][i],word_b_pairs[k][i+1])]==0:
        #                 del pairs_cnt[(word_b_pairs[k][i],word_b_pairs[k][i+1])]
        #         word_b_pairs[k]=replace_pair(word_b_pairs[k],merge_pair,cnt-1)

        #         for i in range(len(word_b_pairs[k])-1):
        #             if (word_b_pairs[k][i],word_b_pairs[k][i+1]) in pairs_cnt: 
        #                 pairs_cnt[(word_b_pairs[k][i],word_b_pairs[k][i+1])]+=word_cnt[k]
        #             else:
        #                 pairs_cnt[(word_b_pairs[k][i],word_b_pairs[k][i+1])]=word_cnt[k]

        pbar.update(1)
    print("get_max_time: ",get_max_time)
    return vocab,merges