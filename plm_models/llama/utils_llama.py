from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.utils import ModelOutput


PLM_MAX_LENGTH = 512


@dataclass
class MyCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    sequence_output: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
