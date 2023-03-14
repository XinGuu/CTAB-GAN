# Adapt CTAB-GAN+ to other raw data format
* Add a `run_ctabgan.py` python file
* Use Opacus to replace the original rdp_accountant

## Install
1. Create a Conda environment using Python 3.7
    ```
    conda create -n ctab python=3.7
    conda activate ctab
    ```
2. Upgrade pip
    ```
    pip install --upgrade pip
    ```
3. Install required packages via setuptools
    ```
    pip install -e .
    ```
4. Install PyTorch 1.9.1 with the appropriate CUDA verison. For example,
    ```
    pip install torch==1.9.1+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
    ```

## Run
* Raw data are under `datasets/raw` folder.
* Under `run` folder, run
    ```
    python run_ctabgan.py --dataset adult --num_exp 1
    ```
  see `run_ctabgan.py` for more arguments.