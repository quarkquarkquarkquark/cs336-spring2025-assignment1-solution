from dataclasses import dataclass
class config:
    
    input_path : "./data/owt_train.txt"
    vocab_filepath : "./tokenizer_files/owt_vocab.pkl"
    merges_filepath : "./tokenizer_files/owt_merges.pkl"
    special_tokens : ["<|endoftext|>"]
    vocab_size : 30000

@dataclass
class train_config:
    input_path : str = "./data/ts_train_tok.bin" 
    eval_path : str = "./data/ts_valid_tok.bin" 
    log_dir : str="./logs/"
    output_dir : str = "./ckpt"
    batch_size : int = 256
    context_length : int=256

    device : str = "cuda"

    vocab_size : int = 10000
    d_model : int = 512
    d_ff : int = 1344
    num_layers : int = 4
    num_heads : int = 16
    rope_theta : float = 10000.0

    lr : float = 4e-3
    betas : tuple = (0.9,0.95)
    training_steps : int = 5000
    eval_steps : int=100
    warmup_iters : int = 200
    cosine_cycle_iters : int = 4500
    
    