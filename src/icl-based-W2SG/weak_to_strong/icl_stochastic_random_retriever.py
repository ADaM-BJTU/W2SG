'''Random Retriever'''

from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger
from typing import List, Union, Optional
from tqdm import trange
import numpy as np
from accelerate import Accelerator

logger = get_logger(__name__)


class StochasticRandomRetriever(BaseRetriever):
    """Random In-context Learning Retriever Class
        Class of Random Retriever.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        seed (`int`, optional): Seed for the random number generator.
    """

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 seed: Optional[int] = 43,
                 accelerator: Optional[Accelerator] = None,**kwargs
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.seed = seed

    def retrieve(self,label_column, max_every_class=None ):
        """
        max_every_class = {0:6, 1:6}
        """
        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        rtr_idx_list = [] 
        label_set = set(self.index_ds["hard_label"])  
        max_every_class ={ label: self.ice_num // len(label_set) +1 for label in label_set} if max_every_class is None else max_every_class # 平均
        logger.info("Retrieving data for test set...")
        for _ in trange(len(self.test_ds), disable=not self.is_main_process):
            idx_list = np.random.choice(num_idx,num_idx , replace=False).tolist()
            new_idx_list = [] 
            ctx_class2num = {label:0 for label in label_set}
            for idx in idx_list:
                idx_label = self.index_ds[idx]["hard_label"]
                if ctx_class2num[idx_label] < max_every_class[idx_label]:
                    ctx_class2num[idx_label] += 1
                    new_idx_list.append(idx)
                    if len(new_idx_list) == self.ice_num:
                        break 
                else: 
                    continue
            rtr_idx_list.append(new_idx_list)
        return rtr_idx_list
