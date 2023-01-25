import argparse
import gc
import itertools
import numpy as np
import os
import pandas as pd
import shutil
import sleap
import tensorflow as tf
import time
from sleap.nn.inference import load_model
from time import perf_counter

sleap.disable_preallocation()


def log_result(data, filename="quadro_p6000_loading_benchmark.csv"):
    df = pd.DataFrame({k: [v] for k, v in data.items()})

    if os.path.exists(filename):
        df = pd.concat(
            [
                pd.read_csv(filename),
                df,
            ]
        )
    df.to_csv(filename, index=False)


def benchmark(converted_path, data, predictor, **kwargs):

    # convert model and load here for more accurate latency readout
    if kwargs["tensor_rt"] != None:
        predictor.convert_model(
            converted_path.replace("pb_2", "pb_1"), converted_path, kwargs["tensor_rt"]
        )

    # warmup run
    predictor.predict(data, **kwargs)

    # real run
    t0 = perf_counter()
    predictor.predict(data, **kwargs)
    dt = perf_counter() - t0

    N = len(data)
    latency = (dt * 1000) / N
    fps = N / dt
    print(f"   Real run: {latency:.2f} ms/img -> {fps:.2f} FPS")

    return dt, latency, fps


def main(labels, model_file, batch_size, opt):

    predictor = load_model(model_file, resize_input_layer=False)
    predictor.batch_size = batch_size

    # ensure we convert with correct precision / batch size
    converted_path = os.path.join(model_file[-1], "pb_2")

    total_time, latency, fps = benchmark(
        converted_path, labels, predictor, tensor_rt=opt, make_labels=False
    )

    opt_str = "None" if opt == None else opt

    log_result(
        {
            "precision": opt_str,
            "batch_size": batch_size,
            "total_time": total_time,
            "latency": latency,
            "fps": fps,
            "n_imgs": len(labels),
        }
    )

    # gives us some overhead with memory growth...
    gc.collect()
    time.sleep(5)


if __name__ == "__main__":

    base = "test_data"

    labels_files = [
        "190719_090330_wt_18159206_rig1.2@15000-17560.slp",
    ]

    model_files = [
        "centroid.fast.210504_182918.centroid.n=1800",
        "td_fast.210505_012601.centered_instance.n=1800",
    ]

    model_files = [os.path.join(base, f) for f in model_files]

    predictor = load_model(model_files, resize_input_layer=False)

    hp = {
        "num_frames": [128, 256, 512, 1024],
        "batch_sizes": [4, 16, 32],
        "opts": [None, "FP32"],
        "labels_files": labels_files,
    }
    combs = list(itertools.product(*hp.values()))

    for comb in combs:

        num_frames, batch_size, opt, labels_file = (comb[0], comb[1], comb[2], comb[3])

        labels = sleap.load_file(os.path.join(base, labels_file))
        labels.labeled_frames = labels.labeled_frames[:num_frames]

        predictor.batch_size = batch_size

        main(labels, model_files, batch_size, opt)
