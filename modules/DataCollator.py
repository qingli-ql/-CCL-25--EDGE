from transformers import DataCollatorWithPadding
import torch

class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, padding=True):
        super().__init__(tokenizer, padding)
    def __call__(self, features):
        types = [feature.pop("type") for feature in features]
        features = [{"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"], 'label':feature['label']} for feature in features]
        batch = super().__call__(features)
        batch["type"] = torch.tensor(types)
        return batch