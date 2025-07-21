# import os
# from tests.adapters import find_max_pairs,get_word_dict,replace_pair,run_train_bpe
# from tests.common import FIXTURES_PATH,gpt2_bytes_to_unicode
# import regex as  re
# import pickle
# import sys
# import time
# from tqdm import tqdm
# import pickle

# if __name__=="__main__":

#     input_path="data/TinyStoriesV2-GPT4-train.txt"
#     t=time.time()
#     vocab, merges = run_train_bpe(
#         input_path=input_path,
#         vocab_size=10000,
#         special_tokens=["<|endoftext|>"],
#     )
#     print(time.time()-t)
#     with open("./ts_vocab.pkl","wb") as f:
#         pickle.dump(vocab,f)

#     with open("./ts_merges.pkl","wb") as f:
#         pickle.dump(merges,f)

#     max_len=0
#     max_word=""
#     for item in vocab.values():
#         if len(item)>max_len:
#             max_len=len(item)
#             max_word=item
#     print(max_word)

#     # # Check that the special token is not in the vocab
#     # vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
#     # for word_bytes in vocabs_without_specials:
#     #     assert b"<|" not in word_bytes


    