# from tests.adapters import Tokenizer
# import pickle
# import regex as re
# import time
# from tqdm import tqdm


# special_tokens=["<|endoftext|>"]
# vocab_filepath="./owt_vocab.pkl"
# merges_filepath="./owt_merges.pkl"

# tokenizer = Tokenizer.from_files(
#     vocab_filepath=vocab_filepath,
#     merges_filepath=merges_filepath,
#     special_tokens=special_tokens
# )

# with open("./data/owt_train.txt") as fin, \
#     open("./data/owt_train_tok.txt","w") as fout:

#     for id in tqdm(tokenizer.encode_iterable(fin)):
#         fout.write(f"{id} ")
