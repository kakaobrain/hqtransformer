## Experiment Command

### Dataset
Set-up root directories for ImageNet, Conceptual Caption 3M, 12M and FFHQ in the [hqvae/datasets/__init__.py](../hqvae/datasets/__init__.py)

### Stage1 HQ-VAE

#### Train
```bash
python main_stage1.py -c=[CONFIG_FILE_LOCATION] -r=[RESULT_CHECKPOINT_PATH] --n-gpus=[NUMBER_OF_GPUS]
```

For example,
```bash
python main_stage1.py -c=configs/master/stage1/imagenet/hqvae-pixelshuffle-top8x8-epoch15.yaml -r=result-stage1 --n-gpus=4
```

#### Evaluation
```bash
python eval_stage1.py -r=[RESULT_CHECKPOINT_PATH] -d=[imagenet, cc15m, ffhq] --code-usage --fid --use-full-checkpoint
```
Note that set up the option **--use-full-checkpoint** when evaluating with downloaded checkpoint of the two-stage model.

### Stage2 HQ-Transformer

#### Train
We do not provide stage 2 training script to avoid unexpected usage of our methods. Note that any commercial use of our checkpoints is prohibited.

#### Evaluation
The checkpoints of HQ-VAE (stage1) and HQ-Transformer (stage2) are provided for reproducing the reported performances.

First, download the statistics and feature for evaluation in the directiory *assets*.
[evaluation_asset.tar.gz](https://www.dropbox.com/s/4le4uudq7cgkkjr/evaluation_asset.tar.gz?dl=0)

Second, sample the images from the model and evaluate generated images.

For class-conditional generation in ImageNet,
```bash
python sampling_hqmodel.py -r=[SAMPLING_PATH] -m=[CHECKPOINT_PATH] --top-k [TOP_K] --temperature [Temperature] && python eval_hqmodel.py -r=[SAMPLING_PATH]

```

For unconditional generation in FFHQ, 
```bash
python sampling_hqmodel.py -r=[SAMPLING_PATH] -m=[CHECKPOINT_PATH] --top-k [TOP_K] --temperature [Temperature] --num_classes 1 && python eval_hqmodel.py -d=ffhq -r=[SAMPLING_PATH]

```

For text-conditional generation in CC-3M
```bash
python sampling_hqmodel_txt2img.py -m=[CHECKPOINT_PATH] -r=[SAMPLING_PATH] --temperature [TEMPERATURE] --top-k [TOP_K] && && python eval_hqmodel.py -d=cc3m -m='fid' -r=[SAMPLING_PATH]
```

If the code resolution is not 8x8 + 16x16, you need to specify it.
The model with code resolution 4x4 + 8x8 + 16x16 in ImageNet can be evaluated by below command.
```bash
python sampling_hqmodel.py -r=[SAMPLING_PATH] -m=[CHECKPOINT_PATH] --top-k 2048 --temperature 1.0 --temperature_decay 0.8 --batch-size 50 --top-resolution 4 --code-level 3 && python eval_hqmodel.py -r=[SAMPLING_PATH]
```

#### Measure throughput
We provide throughput measurement routine for HQ-Transformer. 

A model with code resolution 8x8 + 16x16
```bash
python -m measure_throughput model_path=[CONFIGURATION_FILE_PATH]
```

A model with code resolution 8x8 + 16x16 + 32x32
```bash
python -m measure_throughput model_path=[CONFIGURATION_FILE_PATH] code-level=3
```
