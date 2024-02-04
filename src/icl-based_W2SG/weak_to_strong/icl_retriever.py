# 获得当前文件路径
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
from openicl import  TopkRetriever,  VotekRetriever, ZeroRetriever  
from openicl import BM25Retriever as openicl_BM25Retriever
from openicl.icl_retriever import BaseRetriever
from dataclasses import dataclass
from typing import  Dict   
from weak_to_strong.icl_stochastic_random_retriever import StochasticRandomRetriever
_RETRIEVERS = {
    "topk": TopkRetriever, 
    "vote": VotekRetriever,
    "zero": ZeroRetriever,
    "stochastic_random": StochasticRandomRetriever,  
    "openicl_bm25": openicl_BM25Retriever 
} 
@dataclass
class Retriever:
    retriever_name: str
    ice_num: int
    batch_size: int 
    sentence_transformers_model_name: str 
    tokenizer_name: str 
    ice_separator: str 
    ice_eos_token: str 
    votek_k: int
    cu_num: int
    def __post_init__(self):
        self.retriever = _RETRIEVERS[self.retriever_name] 

    def create_context_retriever(self,data,
                                 ice_num=None,
                                 batch_size=None,
                                 sentence_transformers_model_name=None,
                                 tokenizer_name=None,
                                 sentence_transformers_model=None,
                                 tokenizer=None,
                                 seed=43,
                                 cu_num=None) -> BaseRetriever:  
        return self.retriever(data, 
                              ice_num=ice_num if ice_num else self.ice_num, 
                              batch_size=batch_size if batch_size else self.batch_size,
                              sentence_transformers_model_name=sentence_transformers_model_name if sentence_transformers_model_name else self.sentence_transformers_model_name,
                              tokenizer_name=tokenizer_name if tokenizer_name else self.tokenizer_name,
                              ice_separator=self.ice_separator,
                              ice_eos_token=self.ice_eos_token,
                              votek_k=self.votek_k,
                              sentence_transformers_model=sentence_transformers_model,
                              tokenizer=tokenizer,
                              seed=seed,
                              cu_num=cu_num if cu_num else self.cu_num)

retrievers: Dict[str, Retriever] = {}

def register_retriever(
    name: str, 
    retriever_name: str,
    icl_num: int,
    batch_size: int = 8,
    sentence_transformers_model_name: str = "",
    tokenizer_name: str = "",
    ice_separator: str = '\n',
    ice_eos_token: str = "\n" ,
    votek_k: int = 0,
    cu_num: int = 0
) -> None: 
    retrievers.setdefault(name, {})
    retrievers[name][retriever_name] =  Retriever(
        retriever_name=retriever_name,
        ice_num=icl_num,
        batch_size=batch_size,
        sentence_transformers_model_name=sentence_transformers_model_name,
        tokenizer_name=tokenizer_name,
        ice_separator=ice_separator,
        ice_eos_token=ice_eos_token,
        votek_k=votek_k,
        cu_num=cu_num
    )

register_retriever(
    name= f"{BASE}/sciq/", 
    retriever_name= "topk",
    icl_num= 5,
    batch_size= 8 ,
    sentence_transformers_model_name="model/all-mpnet-base-v2",
    tokenizer_name="model/all-mpnet-base-v2",
)  
register_retriever(
    name= f"{BASE}/sciq/", 
    retriever_name= "openicl_bm25",
    icl_num= 5 
)  
register_retriever(
    name= f"{BASE}/sciq/", 
    retriever_name= "zero",
    icl_num= 0 ,
    ice_eos_token=""
)

register_retriever(
    name= f"{BASE}/sciq/", 
    retriever_name= "vote",
    icl_num= 5,
    batch_size= 8 ,
    sentence_transformers_model_name="all-mpnet-base-v2",
    tokenizer_name="model/all-mpnet-base-v2",
    votek_k=3
)

register_retriever(
    name= f"{BASE}/sciq/", 
    retriever_name= "stochastic_random",
    icl_num= 5, 
) 
def load_retriever(name: str, retriever_name: str) -> Retriever:
    return retrievers[name][retriever_name]