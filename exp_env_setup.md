# 1. Environment Setup

This section describes how to set up the development environment required for running the project.

---

## 1.1. Create and Activate the Conda Environment

    conda create --name py3_10_hf python=3.10
    conda activate py3_10_hf

---

## 1.2. Install PyTorch

Visit the official PyTorch installation page:  
👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

1. Check your **CUDA version** (e.g., 12.8).  
2. Select the proper command at the bottom of the PyTorch installation table.  
   For example, for CUDA 12.8:

        pip install torch torchvision

Then, install **torchaudio**:

    pip install torchaudio

> 💡 **Note:**  
> `torchaudio` depends on the PyTorch version, so make sure to match their versions.

---

## 1.3. Install Additional Dependencies

Install `soundfile` to enable reading **FLAC audio files**:

    pip install soundfile

Install Hugging Face libraries:

    pip install "transformers[torch]" datasets

Install **WebDataset** for dataset streaming:

    pip install webdataset

Install evaluation utilities for speech recognition:

    pip install evaluate jiwer

---

## 1.4. References

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)  
- [Hugging Face Datasets Installation](https://huggingface.co/docs/datasets/installation)
