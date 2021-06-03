# Train FasterRCNN with TLT 3.0

The `faster_rcnn` directory consists of all files you need for training.

```sh
➜ tree faster_rcnn
faster_rcnn
├── faster_rcnn.ipynb
├── faster_rcnn_qat.ipynb
└── specs
    ├── default_spec_resnet18_grayscale.txt
    ├── default_spec_resnet18_retrain_spec.txt
    ├── default_spec_resnet18.txt
    └── frcnn_tfrecords_kitti_trainval.txt
```

The python notebook consists of tlt commands that are used for training.
The `specs` directory consists of specs files for different backbone architectures.

## Download sample data (Optional)

If you're just experimenting, you can download a sample dataset from [here](https://drive.google.com/file/d/1uY_uo9sggKtxeoV9wvNnF9Utac5wVqAa/view?usp=sharing) and place it in `data/` directory.

Your data directory should look like this -

```sh
data
└── training
    ├── images
    └── labels
```

## 0. Open Jupyter notebook

```sh
cd faster_rcnn

jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```

The Jupyter nootebook guides you through the entire training, however some important commands are highlighted here.
Start with `faster_rcnn.ipynb` for regular training and `faster_rcnn_qat.ipynb` for Quantization Aware Training.

## 1. Setting up your workspace

* GPU Index

Set the GPU Index based on the NVIDIA GPU you want to use

```sh
%env GPU_INDEX=0
```

* KEY for model encryption/decrpytion

This key is used to decrpyt your .etlt model

```sh
%env KEY=nvidia_tlt
```

* PATH for local project

```sh
%env LOCAL_PROJECT_DIR=/<path-to-dir>/FasterRCNN-TLT-Triton/
```

## 2. Mounting local drive to docker

This is done to keep all your work stored on your local machine.

```python
# Mapping up the local directories to the TLT docker.
import json
import os
mounts_file = os.path.expanduser("~/.tlt_mounts.json")

# Define the dictionary with the mapped drives
drive_map = {
    "Mounts": [
        # Mapping the data directory
        {
            "source": os.environ["LOCAL_PROJECT_DIR"],
            "destination": "/workspace/tlt-experiments"
        },
        # Mapping the specs directory.
        {
            "source": os.environ["LOCAL_SPECS_DIR"],
            "destination": os.environ["SPECS_DIR"]
        },
    ],
    # set gpu index for tlt-converter
    "Envs": [
        {"variable": "CUDA_VISIBLE_DEVICES", "value": os.getenv("GPU_INDEX")},
    ],
    "DockerOptions":{
        "user": "{}:{}".format(os.getuid(), os.getgid())
    }
}

# Writing the mounts file.
with open(mounts_file, "w") as mfile:
    json.dump(drive_map, mfile, indent=4)
```

## 3. Creating TFRecords

For optimum training, your training data needs to be converted to TFRecords

You can change the specification in `faster_rcnn/specs/frcnn_tfrecords_kitti_trainval.txt'

```json
kitti_config {
  root_directory_path: "/workspace/tlt-experiments/data/training"
  image_dir_name: "images"
  label_dir_name: "labels"
  image_extension: ".jpg"
  partition_mode: "random"
  num_partitions: 2
  val_split: 14
  num_shards: 10
}
image_directory_path: "/workspace/tlt-experiments/data/training"
```

To initiate conversion, the following command is used -

```sh
tlt faster_rcnn dataset_convert --gpu_index $GPU_INDEX -d $SPECS_DIR/frcnn_tfrecords_kitti_trainval.txt \
                     -o $DATA_DOWNLOAD_DIR/tfrecords/kitti_trainval/kitti_trainval
```

## 4. Downloading pretrained weights

NVIDIA NGC provides a variety of FasterRCNN varients for you to choose from

To list available models -

```sh
ngc registry model list "nvidia/tlt_pretrained_object_detection*"
```

To download a model -

```sh
ngc registry model download-version nvidia/tlt_pretrained_object_detection:resnet18
```

## 5. Begin Training

First, configure the training specification by editing the file in `faster_rcnn/specs/default_spec_resnet18.txt`

For an explanation on all parameters, look at the [documentation](https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/object_detection/fasterrcnn.html).

To start training, use the following command -

```sh
tlt faster_rcnn train --gpu_index $GPU_INDEX -e $SPECS_DIR/default_spec_resnet18.txt
```

## 6. Pruning

One advantage of using TLT is model pruning, that helps remove redundant connections.

Set the pruning value experimentally.

```sh
tlt faster_rcnn prune --gpu_index $GPU_INDEX -m $USER_EXPERIMENT_DIR/frcnn_kitti_resnet18.epoch12.tlt \
           -o $USER_EXPERIMENT_DIR/model_1_pruned.tlt  \
           -eq union  \
           -pth 0.2 \
           -k $KEY
```

## 7. Retraining

Just like training, you need to edit your specification, but in `faster_rcnn/specs/default_spec_resnet18_retrain.txt` file

```sh
tlt faster_rcnn train --gpu_index $GPU_INDEX -e $SPECS_DIR/default_spec_resnet18_retrain_spec.txt
```

## 8. Converstion to .etlt deploy format

You can export an encoded model that can be deployed directly with Deepstream or with Triton Inference Server.

* FP32

```sh
tlt faster_rcnn export --gpu_index $GPU_INDEX -m $USER_EXPERIMENT_DIR/frcnn_kitti_resnet18_retrain.epoch12.tlt  \
                        -o $USER_EXPERIMENT_DIR/frcnn_kitti_resnet18_retrain.etlt \
                        -e $SPECS_DIR/default_spec_resnet18_retrain_spec.txt \
                        -k $KEY
```

* FP16

```sh
tlt faster_rcnn export --gpu_index $GPU_INDEX -m $USER_EXPERIMENT_DIR/frcnn_kitti_resnet18_retrain.epoch12.tlt  \
                        -o $USER_EXPERIMENT_DIR/frcnn_kitti_resnet18_retrain_fp16.etlt \
                        -e $SPECS_DIR/default_spec_resnet18_retrain_spec.txt \
                        -k $KEY \
                        --data_type fp16
```

* INT8 with calibration

```sh
tlt faster_rcnn export --gpu_index $GPU_INDEX -m $USER_EXPERIMENT_DIR/frcnn_kitti_resnet18_retrain.epoch12.tlt  \
                        -o $USER_EXPERIMENT_DIR/frcnn_kitti_resnet18_retrain_int8.etlt \
                        -e $SPECS_DIR/default_spec_resnet18_retrain_spec.txt \
                        -k $KEY \
                        --data_type int8 \
                        --batch_size 8 \
                        --batches 10 \
                        --cal_cache_file $USER_EXPERIMENT_DIR/cal.bin
```

