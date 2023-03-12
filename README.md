# Adapt CTAB-GAN+ to other raw data format
* Add a `run_ctabgan.py` python file
* Use Opacus to replace the original rdp_accountant

## Install
1. Create a Conda environment using Python 3.7
    ```
    conda create -n ctab python=3.7
    conda activate ctab
    ```
2. Install required package
   ```
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu111   # You can use other CUDA version
   ```

## Run
* Raw data are under `datasets/raw` folder.
* Under `run` folder, run
    ```
    python run_ctabgam.py --dataset adult --num_exp 1
    ```
  see `run_ctabgan.py` for more arguments.