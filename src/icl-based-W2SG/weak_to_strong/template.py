# 获得当前文件路径
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # W2SG/icl-based-W2SG
from openicl import PromptTemplate
from dataclasses import dataclass
from typing import  Dict, List, Union 
@dataclass
class Template:
    prompt: List[Union[str, Dict[str, str]]]
    system: str 
    tp_dict: Dict[Union[int, str], str] 
    sep_token: str
    def __post_init__(self):
        self.context_template = PromptTemplate(self.tp_dict, {"txt" : '</text>'}, ice_token='</E>'  )
        self.pt_template = PromptTemplate(self.prompt, {'txt' : '</text>'} , ice_token='</E>' )

templates: Dict[str, Template] = {}

def register_template(
    name: str, 
    prompt: List[Union[str, Dict[str, str]]],
    system: str,  
    tp_dict: Dict[Union[int, str], str] ,
    sep_token: str
) -> None:
    template_class =  Template
    templates[name] = template_class( 
        prompt=prompt,
        system=system,  
        tp_dict=tp_dict,
        sep_token=sep_token
    )
    


register_template(
    name= f"{BASE}/sciq/", 
    prompt=  "There is a science knowledge question, followed by an answer. Respond with 1 if the answer is correct, and with 0 otherwise.\n</E></text>\n" ,
    system=(
        ""
    ),  
    tp_dict = {1: "</E></text>",
               0:  "</E></text>"},
    sep_token = "\n"
)

def load_template(name: str) -> Template:
    return templates[name]