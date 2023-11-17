import json
import os
import uuid
from PIL import Image
from django.conf import settings
from paddleocr import PaddleOCR
from tabulate import tabulate

from .layoutMner import LayoutLMNER


def scale_bbox_coordinates(bboxes, image_width, image_height, scaled_min, scaled_max):
    scaled_bboxes = []

    for bbox in bboxes:
        scaled_bbox = [
            int((coord / image_width) * scaled_max) if i % 2 == 0 else int((coord / image_height) * scaled_max)
            for i, coord in enumerate(bbox)
        ]
        scaled_bboxes.append(scaled_bbox)
    return scaled_bboxes


def ocr_and_scale_bboxes(image_path, scaled_min=0, scaled_max=1000):
    img = Image.open(image_path)

    # Obtain OCR results

    ocr = PaddleOCR(lang="fr", use_angle_cls=False)
    ocr_result = ocr.ocr(image_path)

    tokens = []
    bboxes = []

    for item in ocr_result:
        for bbox, (text, confidence) in item:
            # Append token and bbox to respective lists
            tokens.append(text)
            # Convert bbox from [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] to [x1, y1, x2, y2]
            flattened_bbox = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]
            bboxes.append(flattened_bbox)

    # Scale the bboxes
    image_width, image_height = img.size
    scaled_bboxes = scale_bbox_coordinates(bboxes, image_width, image_height, scaled_min, scaled_max)
    return tokens, scaled_bboxes


# change the image path to raw image from post api

def format_for_layoutlm(image_data):
    try:
        # Use PIL to open the image from raw data
        image = Image.open(image_data)
    except Exception as e:
        print(f"Error opening the image: {str(e)}")
        return None

    # Placeholder for ner_tags, replace with actual NER predictions from a pretrained model
    ner_tags = ['O', 'Ref', 'NumFa', 'Fourniss', 'DateFa', 'DateLim', 'TotalHT', 'TVA', 'TotalTTc', 'unitP', 'Qt',
                'TVAP', 'descp']

    # Generate a unique ID for the data
    data_id = uuid.uuid4().hex

    try:
        # Save the image to a file
        temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp')
        image.save(temp_image_path)

        # Convert words and bounding boxes to the desired format
        tokens, bboxes = ocr_and_scale_bboxes(temp_image_path)

        # Save the JSON file
        json_file_path = os.path.join(settings.MEDIA_ROOT, 'temp', f"image_{data_id}.json")
        print(json_file_path)

        # Data in the desired format
        formatted_data = {
            'id': data_id,
            'image': f"image_{data_id}.png",  # Update the image path or use other appropriate naming
            'bboxes': bboxes,
            'ner_tags': ner_tags,
            'tokens': tokens
        }

        # Save the JSON file
        with open(json_file_path, 'w') as f:
            json.dump(formatted_data, f)
            print("Files saved successfully")

        return json_file_path

    except Exception as e:
        print(f"Error processing or saving files: {str(e)}")
        return None


def split_json_data(data):
    # Define the criteria for splitting (e.g., based on the length of words or any other criteria)
    split_point = len(data["tokens"]) // 2  # Splitting based on the number of words

    # Split the data into "top" and "bottom" parts along with their words and bboxes
    top_data = {
        "image": data["image"],
        "id": data["id"],
        "ner_tags": data["ner_tags"],
        "tokens": data["tokens"][:split_point],
        "bboxes": data["bboxes"][:split_point]
    }

    bottom_data = {
        "image": data["image"],
        "id": data["id"],
        "ner_tags": data["ner_tags"],
        "tokens": data["tokens"][split_point:],
        "bboxes": data["bboxes"][split_point:]
    }

    return top_data, bottom_data


def post_process(true_predictions, trimmed_list):
    true_confidence_scores = []
    true_predictions_trimmed = true_predictions[1:-1]
    for idx, pred in enumerate(true_predictions_trimmed):
        true_confidence_scores.append((pred, trimmed_list[idx]))

    return true_confidence_scores, true_predictions_trimmed


def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_results_table(true_predictions_trimmed_par, true_confidence_scores_par, example_par):
    # Create a dictionary to store the word-label pairs and confidence scores
    word_confidence_dict = {}

    for idx, (word, prediction) in enumerate(zip(example_par['tokens'], true_predictions_trimmed_par)):
        if word not in word_confidence_dict and prediction != 'O':
            if prediction == 'O':
                predicted_label = 'other'
            else:
                predicted_label = prediction
            word_confidence_dict[word] = predicted_label

            confidence_score = true_confidence_scores_par[idx][1] if idx < len(true_confidence_scores_par) else 0.0
            word_confidence_dict[word] = {'label': predicted_label.lower(), 'confidence_score': confidence_score}

    # Filter out labels 'other' and 'o'
    filtered_word_confidence_dict = {word: data for word, data in word_confidence_dict.items() if
                                     data['label'] != 'other' and data['label'] != 'o'}

    # Convert the dictionary to a list of tuples for tabulate
    table_data = [(word, data['label'], data['confidence_score']) for word, data in
                  filtered_word_confidence_dict.items()]

    # Define table headers
    headers = ['Word', 'Predicted Label', 'Confidence Score']

    # Return the table using tabulate
    return tabulate(table_data, headers=headers, tablefmt='grid')


def handle_uploaded_file(uploaded_file):
    # Create a temporary file path
    temp_file_path = os.path.join(settings.MEDIA_ROOT, 'temp', uploaded_file.name)

    # Write the uploaded file data to the temporary file
    with open(temp_file_path, 'wb') as temp_file:
        for chunk in uploaded_file.chunks():
            temp_file.write(chunk)

    return temp_file_path


model_path = "ineoApp/LayoutLMv3_5_entities_filtred_14"
layout_ner = LayoutLMNER(model_path)
