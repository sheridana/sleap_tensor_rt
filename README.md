# Overview

We recently had a fun [SLEAP](https://sleap.ai/) hackathon to work on implementing some new features. One of the features we were interested in was having the option to utilize TensorRT for inference optimization on Nvidia GPUs. If you don't have access to a Nvidia GPU then this feature will not work.

[overview](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)

[helful blog](https://blog.tensorflow.org/2021/01/leveraging-tensorflow-tensorrt-integration.html)

[tensorflow api](https://www.tensorflow.org/api_docs/python/tf/experimental/tensorrt/Converter)

[original repo](https://github.com/talmo/gpuhackathon-sleap/tree/main/tensorrt)

[issue](https://github.com/talmolab/sleap/issues/1112)

[pull request](https://github.com/talmolab/sleap/pull/1138)

---

# System / hardware considerations

This feature is currently **extremely** experimental and is likely only useful for experienced users. It was tested on Ubuntu 18 and therefore probably won't work on other versions / systems. If you are running on linux and want to check your version you can run `lsb_release -a`. Example output:

```
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 18.04.6 LTS
Release:	18.04
Codename:	bionic
```

Additionally, it is possible that even if your system requirements are correct, your nvidia gpu might not support conversion. We tested on two different gpus: a Quadro p6000 and an RTX a5000. The quadro was only able to do FP32 conversion, whereas the rtx was able to do both FP32 and FP16 conversion (benchmarks below). We did not test Int8 conversion as it is a bit trickier (needs a calibration function and seems to be less reliable). You can check the Nvidia TensorRT [support matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html) (specifically the hardware section), but even this only lists some examples of supported hardware. If all of these requirements are met, then the following should hopefully work.

---

# Example use

Start by cloning this repo:

```
git clone https://github.com/sheridana/sleap_tensor_rt.git && cd sleap_tensor_rt
```

For this benchmark, we tested a [topdown model](https://sleap.ai/tutorials/initial-training.html?highlight=topdown) on videos containing two / eight flies. We can just download the two fly data and models for now:

```
wget https://www.dropbox.com/s/cggx07j8nwi7wkt/test_data.zip && unzip test_data.zip
```

Getting this to work with SLEAP assuming only conda/pip installations was tricky, as there are many conflicting packages and system requirements. After a lot of trial and error we were able to get it working with a specific TensorRT pypi package and python version. 

To start, ensure that you have [conda](https://docs.conda.io/en/latest/) installed and updated 

Create an environment with python 3.8 and activate it (make sure to run each line sequentially):

```
conda create --name sleap_tensor_rt python=3.8
conda activate sleap_tensor_rt
```

Install relevant TensorRT packages:

```
pip install --upgrade setuptools pip
pip install nvidia-pyindex
pip install nvidia-tensorrt==7.2.3.4
```

Clone SLEAP:

```
git clone https://github.com/talmolab/sleap && cd sleap
```

Checkout remote experimental branch:

```
git checkout origin/arlo/tensor-rt
```

**Make sure to comment out `python=3.7` in `sleap/environment.yml`!!!!**

Updating the activated environment with SLEAP packages sometimes messed with the TensorRT packages. To be safe, deactivate the environment, update it with the yaml, and reactivate it:

```
conda deactivate
conda env update --name sleap_tensor_rt --file environment.yml --prune
conda activate sleap_tensor_rt
```

TensorFlow versions < 2.7 correctly detected the conda-installed CUDA libraries. Newer installations however sometimes have problems with this when installing on linux. E.g sometimes when importing sleap we get a warning `libcudart.so.11.0: cannot open shared object file: No such file or directory`. This means that tensorflow can't figure out how to connect to cuda and will therefore not find your gpu.

You can also test this with:

```py
import tensorflow as tf
tf.test.is_gpu_available()
```

Which should return `True`, but might return `False`. Since we need to access our GPU for this tensor optimization, we need to make sure this returns `True`. 

To fix, run:

```
conda env config vars set LD_PRELOAD=$CONDA_PREFIX/lib/libcudart.so:$CONDA_PREFIX/lib/libcublas.so:$CONDA_PREFIX/lib/libcublasLt.so:$CONDA_PREFIX/lib/libcufft.so:$CONDA_PREFIX/lib/libcurand.so:$CONDA_PREFIX/lib/libcusolver.so:$CONDA_PREFIX/lib/libcusparse.so:$CONDA_PREFIX/lib/libcudnn.so
```

Then deactivate and reactivate your environment:

```
conda deactivate
conda activate sleap_tensor_rt
```

To start, we can try running `test_save.py` as is:

```py
python test_save.py
```

This file simply loads our data and model files, creates a predictor, and then runs inference over the first 128 frames. Currently we have `tensor_rt` set to `None`. This will skip optimization and just fall back to regular prediction. Possible tensor_rt options are `None`, `"FP32"`, and `"FP16"`. Setting it to a different value will throw an assertion error. Now if we set to `"FP32"`, we will most likely see:

```
Tensorflow has not been built with TensorRT support, reverting to normal prediction
```

This is because we need to configure system paths to contain the TensorRT path (very useful [comment](https://github.com/tensorflow/tensorflow/issues/57679#issuecomment-1249197802) for help in getting this working):

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.8/site-packages/tensorrt/
```

Deactivate, reactivate:

```
conda deactivate
conda activate sleap_tensor_rt
```

Now running the script should first convert your model graph to a TensorRT optimized graph and then run inference. The speedup might not be super noticeable since we are just running on 128 frames, but we benchmarked several things and found that TensorRT conversion does indeed optimize inference speeds when available, see below.

--- 

# Benchmarks

## Benchmark overview

We benchmarked FPS and latency over the following:

* GPUs (Quadro P6000, RTX a5000)
* Engines (None, TensorRT)
* Precisions (FP32, FP16)
* Number of Frames (128, 256, 512, 1024)
* Batch sizes (4, 16, 32)
* Number of animals (two flies, eight flies)

We have included an example benchmark script `benchmark.py`. In reality we also included an 8 flies video, and tested FP16 precision when available (on the RTX gpu), but this should give an idea for how to benchmark the different engines. In this example we first convert the model if using TensorRT before running prediction to get more accurate readouts of inference speed. We then perform a warmup inference run to initialize weights, and then calculate speed on the subsequent inference run. 

Note that inside the call to predictor.predict, we [load](https://github.com/talmolab/sleap/blob/24d067a917c399cdf4ffa8b036fff851f9581c2c/sleap/nn/inference.py#L516) our converted model before running inference. This was leading to inaccurate speed readouts (especially at lower frame counts). For the actual benchmarking we:

1) commented these two lines out (516,517)
2) added a parameter to predict: `converted_model = None`
3) set `self.loaded_model_fn = converted_model` under line 517
4) inside the benchmark function, changed the conversion to:

```py
# convert model and load here for more accurate latency readout
if kwargs["tensor_rt"] != None:
    predictor.convert_model(
        converted_path.replace("pb_2", "pb_1"), converted_path, kwargs["tensor_rt"]
    )

    saved_model_loaded = tf.saved_model.load(converted_path)
    converted_model = saved_model_loaded.signatures["serving_default"]
    kwargs["converted_model"] = converted_model
```

## Benchmark results

### Quadro P6000

* Number of frames impacts speed
* Low frame counts are not accurate when considering inference speeds.
* 1024 frames is probably a safe frame count for testing

![](https://github.com/sheridana/sleap_tensor_rt/blob/main/static/fps_vs_frames.png)

* TensorRT does offer speed increase
* Batch size does not seem to have too much affect once you start optimizing 

![](https://github.com/sheridana/sleap_tensor_rt/blob/main/static/quadro_engine_fps.png)
![](https://github.com/sheridana/sleap_tensor_rt/blob/main/static/quadro_engine_latency.png)

* This optimization is also available at lower frame counts
* But this doesn't consider conversion time
* E.g if you had to run inference over many short videos, converting for each would be harmful, converting once up front would be useful.

![](https://github.com/sheridana/sleap_tensor_rt/blob/main/static/quadro_engine_fps_frames.png)

* Loading the converted model in the predictor makes readout unreliable
* Even though inference is much faster, our readout is noisy because of the loading time.
* This noise decreases at higher frame counts, another reason to not benchmark on low frame counts.

![](https://github.com/sheridana/sleap_tensor_rt/blob/main/static/quadro_engine_fps_loading_frames.png)

### RTX A5000

* Further performance gains are possible on better hardware
* We also get increased optimization when lower precision is available

![](https://github.com/sheridana/sleap_tensor_rt/blob/main/static/rtx_engine_fps.png)
![](https://github.com/sheridana/sleap_tensor_rt/blob/main/static/rtx_engine_latency.png)

### Comparing GPUs

* Hardware matters - optimized models on worse hardware are still slower than normal models on better hardware
* Regardless, if you have the option to use TensorRT you should do it. 
* These gains in performance likely compound in practice - if you have long videos or many videos it makes sense to convert up front.

![](https://github.com/sheridana/sleap_tensor_rt/blob/main/static/gpus_fps.png)
![](https://github.com/sheridana/sleap_tensor_rt/blob/main/static/gpus_latency.png)
