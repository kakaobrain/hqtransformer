## Checkpoints
### Checkpoints for Major Experiments

| Dataset        | Code Resolution  | # layers | # feature dim | # codebook         | Top-k, Temperature     | rFID (stage1) | FID (stage2) |   Link   |
|----------------|:----------------:|:--------:|:-------------:|:------------------:|:----------------------:|:-------------:|:------------:|:---------|
| ImageNet (cIN) | 8x8 + 16x16      | 12       | 1536          | 8192 + 8192        | 2048, 0.95             | 2.61          | 9.36         | [link](https://www.dropbox.com/s/cyojh38f923l342/hqtransformer-layer12-imagenet.tar.gz?dl=0) |
| ImageNet (cIN) | 8x8 + 16x16      | 24       | 1536          | 8192 + 8192        | 2048, 0.95             | 2.61          | 8.46         | [link](https://www.dropbox.com/s/bbsfgovsakbxz1y/hqtransformer-layer24-imagenet.tar.gz?dl=0) |
| ImageNet (cIN) | 8x8 + 16x16      | 42       | 1536          | 8192 + 8192        | 2048, 0.95             | 2.61          | 7.15         | [link](https://www.dropbox.com/s/yxb1dqy9jaas84t/hqtransformer-layer42-imagenet.tar.gz?dl=0) |
| CC-15M         | 8x8 + 16x16      | 12       | 1536          | 8192 + 8192        | 8192, 0.9              | 5.76 (CC3M)   | 12.86        | [link](https://www.dropbox.com/s/4xydfhiscwah9n0/hqtransformer-layer12-cc15m.tar.gz?dl=0) |
| FFHQ           | 8x8 + 16x16      | 24       | 1024          | 8192 + 8192        | 4096, 1.0              | 5.53          | 10.21        | [link](https://www.dropbox.com/s/x0ib9ycd82c8g9u/hqtransformer-layer24-ffhq.tar.gz?dl=0) |
