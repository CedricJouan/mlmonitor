# SPDX-License-Identifier: Apache-2.0
import joblib
import os
import pandas as pd
import logging
import json
from io import StringIO
from utils import read_columns
from keras.models import load_model
import tensorflow as tf

try:
    from sagemaker_inference import decoder

except ModuleNotFoundError as e:
    print(f"running locally : {e}")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    model = load_model(os.path.join(model_dir, "model.h5"))
    text_vectorizer = tf.saved_model.load(os.path.join(model_dir, "text_vectorizer.h5"))
    summary_vectorizer = tf.saved_model.load(os.path.join(model_dir, "summary_vectorizer.h5"))
    return model, text_vectorizer, summary_vectorizer


def predict_fn(input_data, model, text_vectorizer, summary_vectorizer):
    log.info("Called predict_fn ")
    log.info(input_data)
    COLUMNS = read_columns()
    df = pd.DataFrame(input_data, columns=COLUMNS)
    texts = df["text"].tolist()
    max_length = 100
    def pred_seq(text):
        input_vector = text_vectorizer([text])
        # Initialize the summary with the start token
        summary_vector = tf.convert_to_tensor([[1]])
        # Generate the summary step by step
        for _ in range(max_length):
            # Predict the next token
            predictions = model.predict([input_vector, summary_vector])
            # Get the last predicted token as the next token in the summary
            next_token = tf.argmax(predictions[:, -1, :], axis=-1, output_type=tf.int32)
            next_token = tf.expand_dims(next_token, axis=0)
            # Break if end token is predicted
            if tf.reduce_all(next_token == 2):
                break
            # Append the predicted token to the summary
            summary_vector = tf.concat([summary_vector, next_token], axis=-1)
        # Detokenize the summary
        # summary_text = summary_vectorizer.detokenize(summary_vector)
        return summary_vector
    seqs = [pred_seq(text) for text in texts]

    def seq_to_summary(seq):
        vocabulary = text_vectorizer.get_vocabulary()
        index_word = {i: word for i, word in enumerate(vocabulary)}

        def indices_to_text(indices):
            return ' '.join(index_word.get(index, '') for index in indices)

        # Then, modify the summary generation part to convert the summary indices to text
        # Assuming summary_vector contains the indices of the generated summary
        predicted_summary_indices = summary_vectorizer.numpy()[0]  # Convert tensor to numpy array
        predicted_summary_text = indices_to_text(predicted_summary_indices)

        print(predicted_summary_text)

    summaries = [seq_to_summary(seq) for seq in seqs]

    records = [
        {
            "summary": summary,
            "score": 1,
        }
        for summary in summaries
    ]
    log.info("predictions")
    log.info(records)
    return records


def input_fn(request_body, request_content_type):
    """An input_fn that loads a csv"""
    log.info("Called input_fn " + request_content_type)
    log.info(request_body)
    if request_content_type == "text/csv":
        COLUMNS = read_columns()
        data = StringIO(request_body)
        df = pd.read_csv(data, sep=",", header=None, names=COLUMNS)
        log.info("returning input for prediction")
        return df.to_numpy()
    elif request_content_type == "application/json":
        jsondata = json.loads(request_body)
        arr = []
        for jsonitem in jsondata["instances"]:
            log.info(jsonitem["features"])
            arr.append(jsonitem["features"])
        return arr
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        np_array = decoder.decode(request_body, request_content_type)
        return np_array


def output_fn(prediction, content_type):
    log.info(f"output_fn:\n{prediction}")
    return {"predictions": prediction}
