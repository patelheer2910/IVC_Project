# IVC Project: Face Analysis, Masking, and Robust Feature Learning

This repository contains an Image & Video Computing (IVC) project exploring **face representation, masking, and feature robustness** under partial occlusion and controlled perturbations.
The project combines **classical computer vision techniques**, **feature-based analysis**, and **modern learning-based approaches**, evaluated primarily on the **Color FERET dataset**.

The central theme is understanding how **face features behave when visual information is suppressed, altered, or selectively removed**, and how this affects recognition and downstream tasks.

---

## Project Goals

* Study the impact of **masking and occlusion** on facial feature representations
* Compare **classical descriptors** and **deep learning–based methods**
* Analyze **texture, shape, and local patterns** in face images
* Investigate robustness using **neuron zeroing, masking strategies, and fine-tuning**
* Benchmark against **state-of-the-art (SOTA)** methods

This project is exploratory and analytical by design, not a black-box classifier demo.

---

## Dataset

### Color FERET Dataset

* Controlled face images with variations in:

  * Pose
  * Expression
  * Illumination
* Metadata stored in `colorferet_metadata.csv`

Used extensively for:

* Feature extraction experiments
* Occlusion and masking analysis
* Comparative evaluation

---

## Repository Structure

```
IVC_Project/
├── notebooks/
│   ├── Colorferet_EDA.ipynb          # Dataset exploration and statistics
│   ├── Image_Features.ipynb          # Feature extraction experiments
│   ├── LBP_algo.ipynb                # Local Binary Pattern implementation
│   ├── Masks.ipynb                   # Mask creation and application
│   ├── finetune-test.ipynb           # Model fine-tuning experiments
│   ├── sota.ipynb                    # SOTA benchmarking
│   ├── sota_finetuning.ipynb         # SOTA + fine-tuning
│   ├── neuron_zeroing.ipynb          # Neuron ablation analysis
│
├── SNLF_Implementation/              # Sparse Non-Linear Feature experiments
├── SNLF_Final/                       # Final SNLF results
│
├── utils.py                          # Shared utility functions
│
├── images/                           # Sample images, masks, and outputs
│   ├── Masked_Image.jpg
│   ├── Masked_Image_with_FaceMesh.jpg
│   ├── output.jpg
│
├── Technical Report - Architecture.pdf
├── shape_predictor_68_face_landmarks.dat
├── README.md
```

---

## Key Components

### 1. Face Detection & Landmark Extraction

* Uses pre-trained **68-point facial landmark predictor**
* Enables precise masking of facial regions
* Supports geometry-aware occlusion experiments

### 2. Masking & Occlusion

* Manual and algorithmic face masks
* Region-specific suppression (eyes, mouth, lower face, etc.)
* Used to simulate real-world occlusions (e.g., masks, obstructions)

### 3. Feature Extraction

* Local Binary Patterns (LBP)
* Texture and shape-based descriptors
* Comparison of handcrafted features vs learned representations

### 4. SNLF (Sparse Non-Linear Features)

* Custom implementation and evaluation
* Focus on interpretability and robustness
* Compared against standard deep features

### 5. SOTA Models & Fine-Tuning

* Baseline deep models evaluated on masked vs unmasked data
* Fine-tuning experiments to assess recovery of performance
* Analysis of representation collapse and resilience

### 6. Neuron Zeroing

* Selective neuron suppression
* Studies internal representation sensitivity
* Provides insight into feature redundancy and importance

---

## Running the Project

This project is notebook-driven.

### Requirements

* Python 3.9+
* OpenCV
* NumPy, Pandas
* PyTorch / TensorFlow (depending on notebook)
* dlib
* scikit-learn
* matplotlib / seaborn

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure `shape_predictor_68_face_landmarks.dat` is present in the root directory.

---

## How to Navigate the Work

A recommended order:

1. `Colorferet_EDA.ipynb` – understand the dataset
2. `Masks.ipynb` – see how occlusion is generated
3. `LBP_algo.ipynb` – classical feature extraction
4. `Image_Features.ipynb` – feature comparisons
5. `sota.ipynb` – baseline deep models
6. `sota_finetuning.ipynb` – robustness via adaptation
7. `neuron_zeroing.ipynb` – internal representation analysis

The **Technical Report** provides architectural and experimental context.

---

## Results & Observations

* Masking significantly degrades naïve face representations
* Some handcrafted features remain surprisingly resilient
* Fine-tuning recovers performance, but not uniformly across regions
* Neuron-level analysis reveals redundancy and brittle dependencies

These findings reinforce the idea that **robust perception is distributed, not localized**.

---

## Limitations

* Dataset is controlled, not in-the-wild
* Masking strategies are synthetic
* Results are exploratory rather than statistically exhaustive

This is a research probe, not a production-ready system.

---

## Future Work

* Multimodal face analysis (RGB + depth)
* Learned masking policies
* Contrastive robustness objectives
* Cross-dataset generalization

---

## License

For academic and educational use.

---
