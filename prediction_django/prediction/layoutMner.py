import numpy as np
import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000)
    ]


class LayoutLMNER:

    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path).to(self.device)
        self.processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
        self.id2label = self.model.config.id2label

    def predict(self, example):
        if isinstance(example['image'], str):
            image = Image.open(example['image'])

        width, height = image.size

        encoded_inputs = self.processor(image, example['tokens'], boxes=example['bboxes'], return_offsets_mapping=True,
                                        return_tensors="pt", max_length=2048, truncation=True)

        offset_mapping = encoded_inputs.pop('offset_mapping')

        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.to(self.device)

        outputs = self.model(**encoded_inputs)
        predictions_values = outputs.logits.argmax(-1).squeeze().tolist()
        is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0

        true_predictions = [self.id2label[pred] for idx, pred in enumerate(predictions_values) if not is_subword[idx]]
        token_boxes = encoded_inputs.bbox.squeeze().tolist()

        true_boxes_values = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if
                             not is_subword[idx]]

        probabilities = outputs.logits.softmax(dim=-1)
        confidence_scores, _ = probabilities.max(dim=-1)
        trimmed_tensor = confidence_scores[:, 1:-1]  # Remove the first and last columns
        trimmed_list_values = trimmed_tensor.squeeze().tolist()

        return true_predictions, true_boxes_values, trimmed_list_values
