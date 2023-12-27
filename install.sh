##
##
##

# activate submodules
git submodule init
git submodule update

# install dependencies
pip install torch torchvision
pip install matplotlib tqdm scipy
git clone https://github.com/fredzzhang/pocket.git
pip install -e pocket

# check that the installation is successful
python main.py --help