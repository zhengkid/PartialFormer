pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install --editable ./
python setup.py build develop
pip install hydra seaborn omegaconf matplotlib tensorboardX scipy
