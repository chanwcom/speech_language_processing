# Fine-Tuning Wav2Vec 2.0 Model

## 1. Introduction



## 2. Retrieving Skeleton Code






## 3. Preparing the Fine-Tuning Dataset
## Fine-Tuning on the MUSIC PART of the STOP training set.


## Fine-Tuning on the Libri-Speech dataset.



## 4. Performing Fine-Tuning

If **GPU 0** is available, set the following environment variables before running the scripts:

    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0

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
