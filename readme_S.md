# Self2Rolling: Self-supervised denoising using detail-guided mask with updates

In this repository, we provide the implementation of **Self2Rolling**.

Our method introduces an innovative cyclic denoising framework using a detail-guided mask with updates. By combining a classical energy model—the Split Bregman algorithm—with a deep denoising neural network, our method performs cyclic denoising to continuously refine and enhance image quality under self-supervised conditions.

## Requirements

Here is the list of libraries you need to install to execute the code:

* Python 3.6
* Tensorflow-gpu 1.14.0
* keras
* scikit-image
* scipy
* cv2 (opencv for python)

All of them can be installed via `conda` (`anaconda`), e.g.:

```bash
conda install scikit-image
```

## How to Execute Demo

1. **Prepare Datasets**: Download the datasets you want to evaluate and save them in `./testsets/`. The current demo is configured to read testing data from `./testsets/S2S,N2N,NE2NE,255075100/Set9/`.
2. **Run the Code**: Execute the demo code in `demo\\\_denoising.py`.

   * The training process uses the Adam optimizer with a learning rate of `1e-4`.
   * By default, it processes images with Gaussian noise levels of `sigma = 10` and `sigma = 15`.
   * The network trains for a total of `30000` steps, saving the model weights and image predictions every `1000` steps.

```bash
python demo\\\_denoising.py
```

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{song2026self2rolling,
  title={Self2Rolling: Self-supervised denoising using detail-guided mask with updates},
  author={Song, Bolin and Wang, Yiyang and Zhou, Changze},
  journal={Neurocomputing},
  year={2026},
  month={May},
  doi={10.1016/j.neucom.2026.132993}
}
```

