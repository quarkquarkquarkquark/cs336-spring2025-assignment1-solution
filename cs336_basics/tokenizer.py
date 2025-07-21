import pickle
from collections.abc import Iterable,Iterator
from tests.adapters import replace_pair
class Tokenizer():
    def __init__(self,
        vocab:dict[int,bytes],
        merges:list[tuple[bytes,bytes]],
        special_tokens:list[str]|None=None
    ):

        self.vocab=vocab
        self.reverse_vocab={value:key for key,value in vocab.items()}
        self.merges=merges
        self.special_tokens=[item.encode('utf-8') for item in special_tokens]
        self.special_tokens_len=[len(tk) for tk in self.special_tokens]
        

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
    
    def encode(self, text:str)->list[int]:
        """Encode an input text into a sequence of token IDs."""
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
                res.append(self.reverse_vocab[text[i]])
                i+=1

        for item in self.merges:
            merge_pair=(self.reverse_vocab[item[0]],self.reverse_vocab[item[1]])
            rep_idx=self.reverse_vocab[item[0]+item[1]]
            res=replace_pair(res,merge_pair,rep_idx)
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
        