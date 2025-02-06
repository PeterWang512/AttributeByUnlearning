## Data Attribution for Text-to-Image Models by Unlearning Synthesized Images

This folder contains code for the MSCOCO experiments reported in our paper.

### Setup
Install dependencies:
```
sudo apt install p7zip-full
micromamba create -f env.yml -y
micromamba activate attr_mscoco
```
`micromamba` was the package manager used and tested thoroughly for the project. However, feel free to try changing it to other package manager such as `mamba` or `conda`.

### Download dataset and models
This script downloads MSCOCO dataset, models, and test samples, along with precomputed fisher information, image latents, and text embeddings.
```
bash data/download_mscoco.sh
```

### Get training loss from pre-trained model
Before assessing influence, our method requires precomputing training loss from the pre-trained model:
```
python compute_training_loss.py --output_path results/pretrain_loss.npy
```
All experiments are conducted using A100 GPUs. If out-of-memory error is encountered, try reducing the batch size with argument `--batch_size`.

### Assess train image influence
After pre-train model's training loss is precomputed, one can now assess influence with the following script:
```
python compute_influence.py \
    --result_dir results \
    --sample_latent_path data/mscoco/coco17_sample_latents.npy \
    --sample_text_path data/mscoco/coco17_sample_text_embeddings.npy \
    --sample_idx 0 \
    --pretrain_loss_path results/pretrain_loss.npy
```
This script calculates influence of the 0th synthesized sample, and one can change the index via `--sample_idx`. The results will be saved in `results/` folder. Also, make sure `--loss_batch_size` and `--loss_time_sample`, the two hyperparameters for loss calculation, are consistent with the ones used in the previous loss calculation process for pre-trained models. Same set of hyperparameter ensures the noise patterns used in the two loss calculation runs are consistent, which are crucial to the performance. This is already done by the default hyperparameters.

### Preprocessing code
Below are additional code to preprocess data.

To obtain fisher information, run the following code. Fisher information will be stored at `<output_path>`.
```
python compute_fisher.py --output_path results/test_fisher.pt
```

To generate the queries used in our paper, run the following code. Images, latents, and text embeddings will be stored at `<output_folder>`. Use the default arguments to ensure deterministic behavior if the goal is to reproduce our queries.
```
python generate_samples.py --output_folder results/test_sample
```

### Acknowledgements
We thank authors of [JourneyTRAK](https://github.com/MadryLab/journey-TRAK) for sharing their MSCOCO model.
