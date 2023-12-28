##
##
##

# activate submodules
git submodule init
git submodule update

# install dependencies
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib tqdm scipy Pillow
git clone https://github.com/fredzzhang/pocket.git
pip install -e pocket

# check that the installation is successful
python main.py --help