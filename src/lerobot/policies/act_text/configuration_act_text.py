#author: Vipul
from dataclasses import dataclass, field
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.policies import PreTrainedConfig

@PreTrainedConfig.register_subclass("act_text")
@dataclass
class ACTTextConfig(ACTConfig):
    """
    Inherits from the ACTConfig and introduces using langugage
    """
    name: str = "act_text"

    # language-specific knobs (do NOT add to baseline)
    use_language: bool = True
    language_model_name: str = "distilbert-base-uncased"
    tokenizer_max_length: int = 128
    freeze_language_encoder: bool = True
    language_pooling: str = "cls"  # "cls" | "mean"