import os
import sleap
from sleap.nn.inference import load_model

sleap.disable_preallocation()

if __name__ == "__main__":

    base = "test_data"

    labels_file = "190719_090330_wt_18159206_rig1.2@15000-17560.slp"

    model_file = [
        "centroid.fast.210504_182918.centroid.n=1800",
        "td_fast.210505_012601.centered_instance.n=1800",
    ]

    model_file = [os.path.join(base, f) for f in model_file]

    labels = sleap.load_file(os.path.join(base, labels_file))

    predictor = load_model(model_file, resize_input_layer=False)

    labels.labeled_frames = labels.labeled_frames[:128]

    preds = predictor.predict(labels, tensor_rt=None, make_labels=False)
