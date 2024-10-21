# MLP Embedded Inverse Tone Mapping

This repository contains the code for the paper:

**MLP Embedded Inverse Tone Mapping**, accepted at **MM2024** (ACM International Conference on Multimedia 2024).

## Overview

This project implements a novel approach to inverse tone mapping using an MLP-based architecture. The code is designed for training and testing on HDRTV4K datasets, but it can be adapted to other datasets as well.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/pjliu3/MLP_iTM.git
    cd MLP_iTM
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Pretrained Model

We provide a pretrained model on the **HDRTV4K** dataset, which you can find in the `pre_model` folder. You can directly use this model for testing if desired.

## Dataset Setup

To train the model, you need to organize your dataset as follows:

- Place the HDR images in `dataset/train/hdr`.
- Place the SDR images in `dataset/train/sdr`.

Follow the format of the provided JSON example to create your own JSON file that describes the dataset. The JSON file should contain the paths to the paired HDR and SDR images for training.

### Example JSON Format

```json
["abp1_autumnwoods_000.jpg", "abp1_bamboo_000.jpg",..., "uzbek_044.jpg", "uzbek_054.jpg", "uzbek_064.jpg", "venice_006.jpg", "venice_016.jpg", "winter_002.jpg", "winter_012.jpg",]

```

## Pretraining

To start pretraining, ensure your dataset is correctly set up, then run:

```bash
python pretrain.py
```

## Testing

You can use the pretrained model for testing on new images. First, ensure that your test images are placed in the `dataset/test` folder. Then, run the following command:

```bash
python test_ours.py
```

The output results will be saved in the `./results` directory.

## Hardware Used

- **Pretraining**: We used 8 NVIDIA RTX 3090 GPUs for pretraining.
- **Inference**: Inference was tested on a single NVIDIA RTX 4090 GPU.

## Contact

If you have any questions or encounter any issues, feel free to contact us at:

**panjun_liu@mail.ustc.edu.cn**

## Citation

If you find this code useful in your research, please cite our paper:

```
@inproceedings{Liu2024MLP,
  author       = {Panjun Liu and Jiacheng Li and Lizhi Wang and Zheng-Jun Zha and Zhiwei Xiong},
  title        = {MLP Embedded Inverse Tone Mapping},
  booktitle    = {Proceedings of the 32nd ACM International Conference on Multimedia (MM '24)},
  year         = {2024},
  location     = {Melbourne, VIC, Australia},
  pages        = {9 pages},
  publisher    = {ACM},
  address      = {New York, NY, USA},
  doi          = {https://doi.org/10.1145/3664647.3680937},
  month        = {October 28-November 1}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.