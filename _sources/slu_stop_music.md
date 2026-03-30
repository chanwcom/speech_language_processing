# Building a Spoken Language Understanding System

## 1. Introduction

Spoken Language Understanding (SLU) represents a critical bridge between human auditory communication and machine-actionable intelligence. Traditionally, SLU systems have relied on a modular pipeline architecture, where a Speech-to-Text (STT) engine first transcribes audio into text, followed by a Natural Language Understanding (NLU) unit that extracts semantic intent and entities. In this project, we explore the implementation of a robust SLU framework by leveraging state-of-the-art self-supervised learning models and modern transformer architectures.

### 1.1 The Cascaded Pipeline Approach
Our primary implementation follows a high-performance cascaded strategy, which allows for independent optimization of the acoustic and semantic components:

Acoustic Modeling with wav2vec 2.0: We utilize wav2vec 2.0 (Baevski et al., 2020), a framework for self-supervised learning of speech representations. By fine-tuning a pre-trained wav2vec 2.0 model on labeled speech data using Connectionist Temporal Classification (CTC) loss, we build a powerful automatic speech recognition (ASR) engine capable of capturing nuanced phonetic features even with limited supervised data.

Intent Classification with BERT: Once the audio is converted into a textual representation, it is processed by BERT (Bidirectional Encoder Representations from Transformers) (Devlin et al., 2019). We fine-tune BERT for the specific downstream task of user intent classification, enabling the system to understand the underlying goal of the speaker's utterance with high contextual accuracy.

### 1.2 Transitioning to End-to-End SLU
While the cascaded approach is highly effective, it often suffers from "error propagation," where transcription mistakes in the ASR phase lead to catastrophic failures in intent classification. To address this, this project further investigates End-to-End (E2E) SLU models.

We explore an Encoder-Decoder Attention architecture that maps acoustic features directly to semantic labels without generating an explicit intermediate transcript. By utilizing a transformer-based encoder for audio and a cross-attention mechanism within the decoder, the model can learn to prioritize task-relevant acoustic cues, potentially surpassing the limitations of traditional decoupled systems.

### 1.3 Project Scope and Objectives
The goal of this project is to provide a comprehensive guide and implementation for building a modern SLU system. We will cover:


 - Fine-tuning strategies for wav2vec 2.0 and BERT.

 - Integration techniques for the cascaded pipeline.

 - Experimental results comparing the cascaded framework against an E2E Encoder-Decoder approach.



## 2. Fine-Tuning Wav2Vec 2.0 model

### 2.1 Retrieving Skeleton Code 
You may find the skeleton code at 

https://github.com/chanwcom/slp_lab/tree/main/codelab/wav2vec_2p0

You may clone the slp_lab git repository by running the following command:
```
git clone https://github.com/chanwcom/slp_lab.git
```

You are supposed to fill out the code marked by TODO.


### 2.2. Preparing the Fine-Tuning Dataset

We use the music portion of the STOP train set. However, we removed 00011525.wav, since the transcript of it seems to contain an error: "play song TITLE_MEDIA on spotify" You may download the compressed sharded WebDataset from the following directory:

https://drive.google.com/file/d/1myqysY_FkaynOfkORBA5xyw4FRJ_OxuW/view?usp=drive_link

So the total number of utterances is reduced from 11563 to 11562.

Please note that you should decompress tar.gz files only once. We will use 10 sharded *.tar file for training and eval.

For the test set, I randomly chose 300 utterances from test_0/music_test. You may download the compressed sharded WebDataset. https://drive.google.com/file/d/1j2z8xb4V5zTb6ChJafZZp8Gtt61_ma_1/view?usp=drive_link

As before, you should decompress tar.gz files only once. We will use 10 sharded *.tar file for training and eval.


### 2.3. Performing Fine-Tuning

If **GPU 0** is available, set the following environment variables before running the scripts:

```
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0
```

> 💡 **Note:**  
> If a different GPU is available (for example, GPU 1 or GPU 2),  
> replace the value in `CUDA_VISIBLE_DEVICES` with the corresponding GPU ID.  
> For example:
>
>     export CUDA_VISIBLE_DEVICES=1
>
> This ensures that your training script runs on the correct GPU device.

```
 'eval_steps_per_second': 3.887, 'epoch': 211.0}
{'loss': 0.0136, 'grad_norm': 0.2718251645565033, 'learning_rate': 5.066666666666667e-06, 'epoch': 213.0}                     
{'loss': 0.0142, 'grad_norm': 0.5199770331382751, 'learning_rate': 3.4000000000000005e-06, 'epoch': 216.0}                    
{'loss': 0.0128, 'grad_norm': 0.5341504216194153, 'learning_rate': 1.7333333333333334e-06, 'epoch': 219.0}                    
{'loss': 0.013, 'grad_norm': 0.33202001452445984, 'learning_rate': 6.666666666666667e-08, 'epoch': 222.0}                     
{'eval_loss': 0.5652906894683838, 'eval_wer': 0.22662431527693244, 'eval_runtime': 28.2358, 'eval_samples_per_second': 92.79, 'eval_steps_per_second': 3.896, 'epoch': 222.0}
{'train_runtime': 1881.837, 'train_samples_per_second': 34.009, 'train_steps_per_second': 1.063, 'train_loss': 1.1427945377230644, 'epoch': 222.0}
100%|█████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [31:21<00:00,  1.06it/s]
```
