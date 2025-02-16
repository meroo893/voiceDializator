# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:33:19 2025

@author: Mert_Kamber
"""
import whisperx
from dataclass import dataclass


class Base:
    pass    
    
@dataclass
class ModelWrapper():
    device:str
    batch_size:int
    compute_type:str
    model: whisperx.Pipeline
    pass