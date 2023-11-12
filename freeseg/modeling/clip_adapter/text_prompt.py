from typing import List, Tuple

import torch
from torch import nn

from third_party.CLIP import clip


class PromptExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._buffer_init = False
        self.with_trainable_params = False

    def init_buffer(self, clip_model):
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        raise NotImplementedError()


class PredefinedPromptExtractor(PromptExtractor):
    def __init__(self, templates: List[str]):
        super().__init__()
        # self.templates = templates
        self.templates = [
        "a photo of a {}.",
        "This is a photo of a {}",
        "There is a {} in the scene",
        "There is the {} in the scene",
        "a photo of a {} in the scene",
        "a photo of a small {}.",
        "a photo of a medium {}.",
        "a photo of a large {}.",
        "This is a photo of a small {}.",
        "This is a photo of a medium {}.",
        "This is a photo of a large {}.",
        "There is a small {} in the scene.",
        "There is a medium {} in the scene.",
        "There is a large {} in the scene.",
    ]

    def init_task_prompt(self, clip_model):
        self.task_embeddings = None

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        text_features_bucket = []
        for template in self.templates:
            noun_tokens = [clip.tokenize(template.format(noun)) for noun in noun_list]
            target_device = clip_model.text_projection.data.device if torch.is_tensor(clip_model.text_projection) else clip_model.text_projection.weight.device 
            text_inputs = torch.cat(noun_tokens).to(target_device)
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_bucket.append(text_features)
        del text_inputs
        # ensemble by averaging
        text_features = torch.stack(text_features_bucket).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features
