Installed python-3.9.18

Due to lack of support for complex32 in older torch version,
I needed to install torch-2.2.1 with python-3.9/pip-3.9

conda create --name neuraloperator
conda activate neuraloperator
/work/atif/packages/python-3.9.18/bin/pip3.9 install --upgrade torch==2.2.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
/work/atif/packages/python-3.9.18/bin/pip3.9 install neuraloperator matplotlib wandb torch_harmonics h5py

export PYTHONPATH=/work/atif/neuraloperator/

# Error calculation
https://arxiv.org/html/2402.17185v1


