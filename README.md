# Real-Time Underwater Image Enhancement via Frequency-Guided Dual-Path Attention (FGDRA)

> Accepted by ICME 2026

## 📌 Overview

Real-time underwater image enhancement (UIE) is crucial for mobile underwater photography and autonomous robotic systems, where practical deployment typically requires low latency and compact models under constrained computational resources.

We propose a lightweight UIE framework that integrates frequency-aware design into ultra-efficient architectures. Specifically, our method introduces:

- **MBRConv-DCT**: a Multi-Branch Reparameterizable Convolution with fixed DCT priors, which injects structured directional frequency information during training.
- **FGDPA**: a Frequency-Guided Dual-Path Attention module that fuses spatial and spectral representations via a dual-path design for adaptive feature modulation.

Both components are fully compatible with structural re-parameterization:

- The convolution branch introduces **zero additional inference cost** after re-parameterization.
- The attention module incurs only **minimal computational overhead**.

Our model achieves:

- **4.23K parameters**
- **600+ FPS**
- State-of-the-art performance on underwater image enhancement benchmarks

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/LethyZhang/FGDRA.git
cd FGDRA
pip install -r requirements.txt
```

### 2. Training

```bash
python main.py -task train -model_task uie -device cuda
```

⚠️ Notes:

- Replace the dataset path in the config file.
- Set the following options in the config:
  ```python
  type = "original"
  need_slims = false
  ```

### 3. Testing

```bash
python main.py -task test -model_task uie -device cuda
```

### 4. Demo

```bash
python main.py -task demo -model_task uie -device cuda
```

---

## 📂 Dataset

We provide a predefined test split for the UIEB dataset:

- `uieb_test.txt`: contains the test image list  
- The remaining samples are used for training

⚠️ Since UIEB has no officially standardized split, direct evaluation of the released checkpoint on a different split may be affected by protocol mismatch.  
For fair comparison, please retrain the model using the target split.

---

## 📦 Pretrained Model

Pretrained weights are available in:

```
experiments/pretrain/
```

---

## 📊 Results

| Method | Params | FPS   | PSNR | SSIM |
|--------|--------|--------|------|------|
| Ours   | 4.23K | 600+  | 23.97 | 0.9155 |

---

## 📂 Project Structure

```
FGDRA/
├── config/              # Configuration files
├── data/                # Dataset loader
├── experiments/         # Logs and pretrained models
│   └── pretrain/
├── model/               # Network architecture
├── main.py              # Entry point
├── loss.py
├── option.py
├── logger.py
├── uieb_test.txt        # Test split list
```

---

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2026fgdra,
  title={Real-Time Underwater Image Enhancement via Frequency-Guided Dual-Path Attention},
  author={Zhang, Leshen and Li, Ao and Zhu, Ce},
  booktitle={ICME},
  year={2026}
}
```

---

## 👥 Authors

- Leshen Zhang  
  (University of Electronic Science and Technology of China & University of Glasgow) *(Equal contribution)*  
- Ao Li  
  (University of Electronic Science and Technology of China) *(Equal contribution)*  
- Ce Zhu  
  (University of Electronic Science and Technology of China) *(Corresponding author)*  

---

## 📧 Contact

- lethyzhang@163.com  
- lethyacademic@gmail.com

  ---

## 🙏 Acknowledgement

This project is partially built upon:

https://github.com/LukasYan30/MobileIE
