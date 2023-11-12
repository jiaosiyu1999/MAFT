import logging
import torch
from detectron2.utils.logger import log_first_n
from .text_prompt import (
    PredefinedPromptExtractor,
)
from .adapter import ClipAdapter, MaskFormerClipAdapter# , PerPixelClipAdapter


def build_prompt_learner(cfg, ):
    return PredefinedPromptExtractor(cfg.PREDEFINED_PROMPT_TEMPLATES)
