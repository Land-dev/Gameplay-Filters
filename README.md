# Gameplay Filters: Robust Zero-Shot Safety through Adversarial Imagination

## Installation
If you are using Linux, run the following commands on your computer
```bash
bash install_packages.sh
```

Alternatively, if your computer is not running Linux, run the following commands:
```bash
conda create -n gameplay python=3.8 -y
conda activate gameplay
conda install cuda -c nvidia/label/cuda-11.8.0 -y
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia/label/cuda-11.8.0 -y
conda install -c conda-forge suitesparse jupyter notebook omegaconf numpy tqdm gym dill plotly shapely wandb matplotlib pybullet pandas -y
pip install --upgrade jax==0.4.8 jaxlib==0.4.7+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```

## Usage
**Note**: You will need to setup Weights & Biases API key via conda if you want to visualize the training progress.
1. Create a Weight and Biases account
2. Get your `API key` from your account
3. Run the following command with your API key:
```bash
conda env config vars set WANDB_API_KEY=<your api key>
```
Alternatively, you can also run:
```bash
wandb login
```
And paste your API key when prompted.

On the other hand, if you do not want to log your training progress, open the `.yaml` file in `config/` that you will be used for training and set `USE_WANDB` to `False`.

**If you just want to try out the models in evaluation mode, you can ignore this step.**

### Synthesize gameplay filter
Run the following command:
```bash
python3 train/train_isaacs.py -cf config/isaacs_spirit.yaml
```

This command will train a new gameplay filter. The output will be 3 models: the safety fallback policy, the adversarial policy, and the critic. The result will be put in the folder `train_result/OUT_FOLDER` as stated in `OUT_FOLDER` in the config file. The training progress will be logged onto wandb, under the project `PROJECT_NAME` with the name `NAME` as stated in the config file.

### Evaluate the gameplay filter
Run the following command to evaluate the trained model:
```bash
python3 eval/eval_isaacs.py -cf train_result/OUT_FOLDER/RUN/config.yaml --gui
```

## Citation 
The original source code of ISAACS can be found [here](https://github.com/SafeRoboticsLab/ISAACS).

If you find our paper or code useful, please consider citing us with:
```
@inproceedings{nguyen2024gameplayfiltersrobustzeroshot,
      title={Gameplay Filters: Robust Zero-Shot Safety through Adversarial Imagination}, 
      author={Duy P. Nguyen and Kai-Chieh Hsu and Wenhao Yu and Jie Tan and Jaime F. Fisac},
      year={2024},
      eprint={2405.00846},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2405.00846}, 
}
```