import tensorflow as tf
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tracemalloc
import time
import random
from memory_profiler import profile
import argparse

# Set environment variables to use multiple CPU cores
os.environ["TF_NUM_INTEROP_THREADS"] = "0"
os.environ["TF_NUM_INTRAOP_THREADS"] = "0"

IMAGE_DIR = "./imagenet-sample-images"


def _get_image_names(N):
    file_list = [
        f"./{IMAGE_DIR}/{file}"
        for file in os.listdir(IMAGE_DIR)
        if file.lower().endswith((".jpg", ".jpeg"))
    ]
    assert len(file_list) >= N
    # return random.sample(file_list, N)
    return file_list


def _decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [224, 224])


def _process_path(file_path):
    img = tf.io.read_file(file_path)
    img = _decode_img(img)
    return preprocess_input(img)


def _get_data(N):
    img_paths = _get_image_names(N)
    list_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    # Set `num_parallel_calls` to automatically choose the optimal number of threads
    list_ds = list_ds.map(
        _process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    list_ds = list_ds.batch(N)
    return list_ds


def _inference(batch_size=1, total_batches=1):
    dataset = _get_data(batch_size)
    model = tf.keras.applications.ResNet50(weights="imagenet")
    i = 0
    for x in dataset.take(total_batches):
        predictions = model.predict(x)
        i += 1
        # print(decode_predictions(predictions, top=1))


def inference(batch_size=1, total_batches=1):
    _inference(batch_size, total_batches)


@profile
def inference_profiled(batch_size=1, total_batches=1):
    _inference(batch_size, total_batches)


def latency(mem_prof):
    start = time.time()
    if mem_prof:
        inference_profiled()
    else:
        inference()
    end = time.time()
    print(f"\nLatency: {end - start} S")


def throughput(mem_prof):
    batch_size = 64
    total_batches = 4
    start = time.time()
    if mem_prof:
        inference_profiled(batch_size, total_batches)
    else:
        inference(batch_size, total_batches)
    end = time.time()
    t = float(batch_size * total_batches) / (end - start)
    print(f"\nThroughput: {t}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some flags.")
    parser.add_argument(
        "--throughput", "-t", action="store_true", help="Enable throughput processing"
    )
    parser.add_argument(
        "--latency", "-l", action="store_true", help="Enable latency processing"
    )
    parser.add_argument(
        "--memory", "-mem", action="store_true", help="Enable memory profiling"
    )
    args = parser.parse_args()
    return args


def __main__():
    args = parse_arguments()
    if args.throughput:
        throughput(args.memory)
    elif args.latency:
        latency(args.memory)


if __name__ == "__main__":
    __main__()
