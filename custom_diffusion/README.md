## Data Attribution for Text-to-Image Models by Unlearning Synthesized Images

This folder contains code for the custom diffusion experiments reported in our paper. We use the [Attribution by Customization (AbC)](https://github.com/peterwang512/GenDataAttribution) benchmark.

### Setup
Install dependencies:
```
sudo apt install p7zip-full
micromamba create -f env.yml -y
micromamba activate attr_abc
```
`micromamba` was the package manager used and tested thoroughly for the project. However, feel free to try changing it to other package manager such as `mamba` or `conda`.

### Download dataset and models
This script downloads AbC dataset, models, and test samples, along with precomputed fisher information, image latents, and text embeddings.
```
bash data/download_abc.sh
```

### Get training loss from pre-trained model
Before assessing influence, our method requires precomputing training loss from the pre-trained custom diffusion model:
```
python compute_training_loss.py --task_json tasks/demo_task.json
```
This is an example script that run on a demo test case `tasks/demo_task.json`. The full test case used in the paper is in `tasks/all_tasks.json`. Results are stored in `results/` directory by default. All experiments are conducted using A100 GPUs. If out-of-memory error is encountered, try reducing the batch size with argument `--batch_size`. In our paper, we conduct studies on a 100K LAION subset for runtime concerns. To further reduce the runtime, one can change the testing scenario with fewer LAION images via `--laion_subset_size`.

### Assess train image influence
After pre-train model's training loss is precomputed, one can now assess influence with the following script:
```
python compute_influence.py --task_json tasks/demo_task.json
```
This script calculates influence of synthesized samples in the json file. Again, results will be saved in `results/` folder by default. Also, make sure `--loss_batch_size` and `--loss_time_sample`, the two hyperparameters for loss calculation, are consistent with the ones used in the previous loss calculation process for pre-trained models. Same set of hyperparameter ensures the noise patterns used in the two loss calculation runs are consistent, which are crucial to the performance. This is already done by the default hyperparameters.
