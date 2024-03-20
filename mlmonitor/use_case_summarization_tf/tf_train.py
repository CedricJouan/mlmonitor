# SPDX-License-Identifier: Apache-2.0
import logging
import argparse
import json
import time
import os
import numpy as np
from typing import Optional
import tensorflow as tf
import pandas as pd
import joblib

from tensorflow.keras.layers import TextVectorization, Embedding, GRU, Dense
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential


try:
    from tf_models import base_model
    from factsheets_helpers import init_external_fs_client, save_fs_model
    from utils import init_logger
except ImportError as e:
    print(
        f"use_case_summarization_tf.tf_train could not import modules => not running in AWS job : {e}"
    )
    from mlmonitor.use_case_summarization_tf.tf_models import base_model
    from mlmonitor.use_case_summarization_tf.factsheets_helpers import (
        init_external_fs_client,
        save_fs_model,
    )
    from mlmonitor.use_case_summarization_tf.utils import init_logger, generate_data

vocab_size = 10000
sequence_length = 100
log_level = int(os.getenv("LOG_LEVEL", logging.INFO))

logger = init_logger(level=log_level)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_wml(
    model_dir: str,
    data_path: str,
    train_dataset: str,
    val_dataset: Optional[str] = None,
    test_dataset: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **hyperparameters,
) -> str:
    """
    train_wml is used to train this model in local environment
    this python module `use_case_gcr` is imported dynamically by `mlmonitor`

    - this function definition should always contain as input parameters :
    model_dir , data_path , train_dataset,val_dataset,test_dataset ,logger ,and  hyperparameters as parameters

    - this function must produce a model artifact return its location in model_data pah

    .. code-block:: python
        :caption: Example
        from mlmonitor import WMLModelUseCase
        model_use_case = WMLModelUseCase(source_dir='use_case_gcr', catalog_id=catalog_id, model_entry_id=model_entry_id)
        model_use_case.train() => this function is invoked by trained task

    :param model_dir:str: Base directory where to store model after training completion
    :param data_path:str: location (directory path) of the datasets for this model use case
    :param train_dataset:str: filename of training dataset
    :param val_dataset:Optional[str]=None:  filename of validation dataset
    :param test_dataset:Optional[str]=None:  filename of test dataset
    :param logger:Optional[logging.Logger]=None: Pass instantiated logger object
    :param **hyperparameters: model hyperparameters to use for model training task
    :return: path to the model artifact produced
    """

    trained_model = train_loop(
        train_data=data_path,
        test_data=data_path,
        logger=logger,
        **{
            "epochs": hyperparameters.get("epochs"),
            "batch_size": hyperparameters.get("batch-size"),
        },
    )

    model_data = os.path.join(model_dir, "model_mnist", "mnist_cnn_aws.h5")
    trained_model.save(model_data)

    return model_data

def fetch_dataset(data_path: str, filename: str = "training.csv") -> pd.DataFrame:
    # Take the set of files and read them all into a single pandas dataframe
    print(f"fetch_dataset {os.listdir(data_path)}")
    input_files = [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if filename in file
    ]

    if len(input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(data_path, "train")
        )
    raw_data = [pd.read_csv(file, engine="python") for file in input_files]

    return pd.concat(raw_data)

def train_loop(
    train_data: str,
    test_data: str,
    logger: logging.Logger,
    **hyperparameters,
) -> Model:

    verbose = 0 if os.environ.get("SM_TRAINING_ENV") else 1

    train_texts, val_texts, train_summaries, val_summaries = train_test_split(train_data['text'].tolist(), 
                                                                              train_data['summary'].tolist(),
                                                                                test_size=0.2,
                                                                                random_state=42)
    test_texts, test_summaries = test_data['text'].tolist(), test_data['summary'].tolist()

    logger.info(f"x_train shape: {train_data.shape}")
    logger.info(f"x_test shape: {test_data.shape}")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_summaries)).batch(hyperparameters.get("batch_size"))

    # Text vectorization
    text_vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)
    summary_vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)

    # Adapt vectorizers to the training dataset
    text_vectorizer.adapt(train_dataset.map(lambda x, y: x))
    summary_vectorizer.adapt(train_dataset.map(lambda x, y: y))

# Prepare text and summary for training, validation, and testing
    def prepare_data(texts, summaries, vectorizer):
        texts_vectorized = vectorizer(texts)
        summaries_vectorized = vectorizer(summaries)
        return texts_vectorized, summaries_vectorized[:, :-1], summaries_vectorized[:, 1:]

    train_text_vectorized, train_summary_input, train_summary_target = prepare_data(train_texts, train_summaries, text_vectorizer)
    val_text_vectorized, val_summary_input, val_summary_target = prepare_data(val_texts, val_summaries, text_vectorizer)

    model = base_model(vocab_size=vocab_size)
    model.summary()

    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True))
    history = model.fit(x=[train_text_vectorized, train_summary_input], y=train_summary_target, 
                    validation_data=([val_text_vectorized, val_summary_input], val_summary_target),
                    epochs=hyperparameters.get("epochs"),
                    batch_size=hyperparameters.get("batch_size"),
                    verbose=verbose,)
    


    test_text_vectorized, test_summary_input, test_summary_target = prepare_data(test_texts, test_summaries, text_vectorizer)
    score = model.evaluate(x=[test_text_vectorized, test_summary_input], y=test_summary_target)
    logger.info(f"Cross Entropy: {score}")

    return model, text_vectorizer, summary_vectorizer

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    # fmt: off
    # CP4D specific arguments
    parser.add_argument("--catalog-id", type=str)
    parser.add_argument("--model-entry-id", type=str)
    parser.add_argument("--ibm-key-name", type=str, default="IBM_API_KEY_MLOPS")
    parser.add_argument("--cp4d-env", type=str, default=os.getenv("ENV", "saas"), choices=["saas", "prem"])
    parser.add_argument("--cp4d-username", type=str, default=None)
    parser.add_argument("--cp4d-url", type=str, default=None)

    # Training Job specific arguments (Sagemaker,Azure,WML) default SageMaker envar or Azure expected values
    parser.add_argument("--model-name", type=str, default='summarizer.h5')
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "./outputs"))

    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.getenv("SM_CHANNEL_TEST"))
    parser.add_argument("--validation", type=str, default=os.getenv("SM_CHANNEL_VALIDATION"))

    parser.add_argument("--hosts", type=list, default=json.loads(os.getenv("SM_HOSTS", '["algo-1"]')))
    parser.add_argument("--current-host", type=str, default=os.getenv("SM_CURRENT_HOST", "algo-1"))
    parser.add_argument("--region-name", type=str, default="ca-central-1")

    # Model specific hyperparameters
    parser.add_argument("--batch-size", type=int, default=128, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=3, metavar="N", help="number of epochs to train (default: 1)")
    # fmt: on
    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    args = vars(args)

    log_level = int(os.getenv("LOG_LEVEL", logging.INFO))
    logger = init_logger(level=log_level)
    (
        facts_client,
        props,
        EXPERIMENT_NAME,
        EXPERIMENT_ID,
        tags,
        params,
    ) = init_external_fs_client(
        logger=logger,
        ibm_key_name=args.get("ibm_key_name"),
        region_name=args.get("region_name"),
        catalog_id=args.get("catalog_id"),
        model_entry_id=args.get("model_entry_id"),
        cp4d_env=args.get("cp4d_env"),
        cp4d_username=args.get("cp4d_username"),
        cp4d_url=args.get("cp4d_url"),
    )

    start = time.time()
    print(f'training {os.environ.get("SM_CHANNEL_TRAINING")}')
    print(f'train {os.environ.get("SM_CHANNEL_TRAIN")}')
    print(f"params {params}")
    print(f'SM_CHANNEL_TRAIN {os.environ.get("SM_CHANNEL_TRAIN")}')
    print(f'SM_CHANNEL_TRAINING {os.environ.get("SM_CHANNEL_TRAINING")}')
    train_data = fetch_dataset(data_path=args.get("train"), filename="training.csv")
    test_data = fetch_dataset(data_path=args.get("train"), filename="testing.csv")

    trained_model, text_vectorizer, summary_vectorizer = train_loop(
        train_data=train_data,
        test_data=test_data,
        logger=logger,
        **{"epochs": args.get("epochs"), "batch_size": args.get("batch_size")},
    )

    trained_model.save(os.path.join(args.get("model_dir"), "model.h5"))
    tf.saved_model.save(text_vectorizer, os.path.join(args.get("model_dir"), "text_vectorizer"))
    tf.saved_model.save(summary_vectorizer, os.path.join(args.get("model_dir"), "summary_vectorizer"))

    end = time.time()
    metrics = {"train_duration_sec": np.round(end - start, 4)}

    save_fs_model(
        logger=logger,
        facts_client=facts_client,
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        catalog_id=args.get("catalog_id"),
        model_entry_id=args.get("model_entry_id"),
        tags=tags,
        params=params,
        metrics=metrics
    )