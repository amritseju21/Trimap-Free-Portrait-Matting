# Trimap-Free Portrait Matting

E9 241 (Digital Image Processing) final project: a lightweight, trimap-free
U-Net for human portrait matting, benchmarked against Closed-Form matting,
KNN matting, and ViT-Matte.

**Team:** Rishi Gupta (24504), Amrit Lal (24608)

## Overview

Image matting separates a foreground subject from its background, producing
a soft alpha matte rather than a hard segmentation mask. It's used for
compositing, visual effects, and as a preprocessing step for tasks like face
recognition.

Most matting methods (Closed-Form, KNN, ViT-Matte) need a **trimap** — a
manually or semi-automatically drawn map marking definite foreground,
definite background, and an "unsure" region — as input. Trimaps aren't
available in real-time or large-scale deployments, and state-of-the-art
matting models are often too heavy for fast inference.

This project trains a **lightweight U-Net that predicts a matte directly
from an RGB image, with no trimap**, and compares it against training-free
alpha matting and a pretrained vision-transformer matting model to see how
much accuracy is sacrificed for that convenience.

## Dataset

We use the **Human Matting Dataset** — 34,425 image/binary-mask pairs of
human portraits — split into 27,540 train and 6,885 test samples (the same
test split is used for every method).

Since this dataset ships without trimaps, we synthetically generate them for
the trimap-dependent baselines: erode and dilate the ground-truth binary
mask, then XOR the two results to get the "unsure" gray region, with the
remaining pixels set to match the binary mask.

## Methodology

**Model.** A U-Net (originally built for biomedical image segmentation) is
well suited to matting since both are dense, pixel-wise prediction tasks. To
keep it lightweight, every convolutional layer's channel width is cut to 1/4
of the original U-Net, giving a small encoder-decoder with skip connections
that takes an RGB image in and outputs a single-channel matte.

**Training.** Trained from scratch on the Human Matting train split for 50
epochs, batch size 128, learning rate 1e-5, using BCE loss and the Adam
optimizer. Training took ~4 hours on two 24GB RTX A5000 GPUs.

**Baselines compared:**
- **Closed-Form Matting** — classical, training-free alpha matting from a
  trimap.
- **KNN Matting** — another training-free, trimap-based classical method.
- **ViT-Matte** — a pretrained, state-of-the-art vision-transformer matting
  model (also trimap-based).
- **Custom U-Net** — our trimap-free model.

**Evaluation.** Every method is scored on the common test split using MSE,
SAD (sum of absolute differences), pixel-wise accuracy, and gradient error
(a perceptual matting metric), along with per-sample inference time.

**Extending to multi-face scenes.** Since the U-Net is trained on
single-portrait images, it generalizes poorly to out-of-domain scenes with
multiple people. To handle that case without matting the whole (larger,
noisier) image, we run Viola-Jones face detection first, crop a padded
bounding box around each detected face, and run the U-Net on each crop
independently — producing a per-person matte at low computational cost. This
is useful for downstream tasks like facial recognition or video
surveillance in scenes with multiple people.

## Outcome

| Method | MSE | SAD | Accuracy | Gradient | Inference Time |
|---|---|---|---|---|---|
| Closed-Form Matting | 0.1719 | 10643 | 96.42 | 60.88 | 330.71 ms |
| KNN Matting | 0.429 | 26687 | 96.43 | 47.50 | 3635.75 ms |
| ViT-Matte | 0.0316 | **2432** | 95.88 | 102.84 | 314.82 ms |
| **U-Net (Trimap-free)** | **0.0245** | 2633 | **96.48** | **41.09** | **3.54 ms** |

- Despite being the only trimap-free method, the U-Net **beats ViT-Matte and
  Closed-Form matting on MSE, accuracy, and gradient error**, and is close
  to ViT-Matte on SAD — largely because it's trained directly on
  in-domain data, while the others rely on generic priors or pretraining.
- The U-Net is **~90x faster than ViT-Matte and ~1000x faster than KNN
  matting** per sample (3.54 ms vs. 314.82 ms / 3635.75 ms), making it
  practical for real-time use on a GPU.
- The trade-off is **generalization**: the U-Net struggles on out-of-domain
  images (e.g., multi-person scenes, unusual backgrounds/lighting) that
  differ from its single-portrait training distribution. The Viola-Jones +
  U-Net face-extraction pipeline mitigates this for the multi-face case
  while keeping inference fast.
- Open questions for future work: whether fine-tuning ViT-Matte closes the
  gap, whether lighter transformer segmentation models (e.g., TopFormer)
  beat the U-Net on speed/accuracy, and how much smaller/shallower the U-Net
  can go.

## Repository Structure

```
UNet.py                     U-Net architecture (width-reduced, 4 down/up blocks)
train.py                    Training loop (BCE loss, Adam, 50 epochs)
dataloader.py                Dataset classes (HumanMattingDataset, Alpha_Dataset, ViT_Dataset)
trimaps.py                   Synthetic trimap generation via erosion/dilation
utils.py                     Metrics (MSE, SAD, accuracy, gradient error) + Viola-Jones face crop
eval/
  evaluate_unet.py           U-Net benchmark (metrics + inference time)
  evaluate_alpha.py          Closed-Form / KNN matting benchmark
  evaluate_ViTMatte.py       ViT-Matte benchmark
  demo.ipynb                 Evaluation walkthrough
pipeline_demo.ipynb          Viola-Jones + U-Net multi-face extraction demo
checkpoints/                 Trained model weights
results/                     Qualitative output samples
```

## Setup

```bash
pip install torch torchvision opencv-python pymatting transformers \
            scikit-image scipy tqdm matplotlib pillow
```

Expects a dataset laid out as `dataset/{train,test}/{img,mask}`.

## Usage

```bash
# Generate synthetic trimaps from binary masks (needed for the alpha-matting baselines)
python trimaps.py

# Train the U-Net
python train.py

# Benchmark each method against the test split
python eval/evaluate_unet.py
python eval/evaluate_alpha.py
python eval/evaluate_ViTMatte.py
```

See `pipeline_demo.ipynb` for the Viola-Jones + U-Net pipeline that extracts
per-face mattes from a multi-person image.

## References

Full details and citations (Closed-Form matting, KNN matting, ViT-Matte,
U-Net, and the perceptual matting benchmark used for the gradient-error
metric) are in the project report, `E9 241: Trimap-free Portrait Matting`.
