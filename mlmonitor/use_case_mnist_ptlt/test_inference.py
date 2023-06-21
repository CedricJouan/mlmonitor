# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import numpy as np
import random
import matplotlib.pyplot as plt

from utils import mnist_to_numpy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-type",
    type=str,
    default="pt-lt",
    choices=["cnn", "fc", "pytorch", "tf-cnn"],
    metavar="MDLTYPE",
    help="type of model to run inference",
)
parser.add_argument(
    "--inference-samples",
    type=int,
    default=2,
    metavar="NSAMPLES",
    help="Number of samples to be sent for inference",
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running Inference for Pytorch Model {args}")
    data_dir = "/tmp/data"
    model_type = args.model_type
    print(f"model type {model_type}")

    X, Y = mnist_to_numpy(data_dir, train=False)

    # randomly sample 16 images to inspect
    mask = random.sample(range(X.shape[0]), args.inference_samples)
    samples = X[mask]
    labels = Y[mask]
    # plot the images
    fig, axs = plt.subplots(nrows=1, ncols=args.inference_samples, figsize=(16, 1))

    for i, ax in enumerate(axs):
        ax.imshow(samples[i])
    plt.show()

    if model_type not in ["cnn", "fc"]:
        raise ValueError("model type should be set to cnn or fc")

    from use_case_mnist_ptlt.ptlt_inference import (
        model_fn,
        input_fn,
        predict_fn,
        output_fn,
    )

    samples = np.expand_dims(samples, axis=1)
    inputs = {"input_data": [{"values": samples.tolist()}]}
    model = model_fn("../models")
    print(samples.shape, samples.dtype)
    print(json.dumps(inputs))
    input_tensors = input_fn(json.dumps(inputs), "application/json")
    print(input_tensors.shape)
    outputs = predict_fn(input_tensors, model)
    print(outputs.shape)
    preds = output_fn(outputs, "application/json")
    print(preds)

    predictions = np.argmax(
        np.array(json.loads(preds), dtype=np.float32), axis=1
    ).tolist()
    print("Predicted digits: ", predictions)
