# lerobot/policies/act_text/processor.py
from typing import Any, Dict, List, Tuple
import torch
from transformers import AutoTokenizer

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    ProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


from lerobot.policies.act_text.configuration_act_text import ACTTextConfig


class LanguageTokenizerProcessorStep(ProcessorStep):
    """
    Tokenizes a per-sample instruction string into input_ids + attention_mask.
    Looks for observation.task (fallback: observation.instruction / observation.language_task).
    """
    def __init__(self, model_name: str, max_length: int):
        super.__init__(name="language_tokenizer")
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        instr: List[str] = (
            batch.get("task_index")
        )
        if instr is None:
            raise KeyError(
                "Language-guided ACT expects observation.task (or instruction/language_task) per sample."
            )

        enc = self.tok(
            instr,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch["language_tokens"] = enc["input_ids"].to(dtype=torch.long)           
        batch["language_attention_mask"] = enc["attention_mask"].to(dtype=torch.long)       
        
        return batch


def make_act_text_pre_post_processors(
    config: ACTTextConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> Tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Pre/post pipelines for act_text.
    Order:
      RenameObservations -> AddBatchDimension -> LanguageTokenizer -> Device -> Normalizer
    """
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        LanguageTokenizerProcessorStep(
            model_name=config.language_model_name,
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
