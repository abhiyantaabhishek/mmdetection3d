3D Vehicle Detection (mmdet based)

## Older process to set mmdetection (2021)

conda deactivate
conda env remove --name abhi
conda create --name abhi python=3.7 -y
conda activate abhi
pip3 install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
pip install openmim
mim install mmcv-full==1.5.2
mim install mmdet
mim install mmsegmentation
rm -r -f mmdetection3d/
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -e .
mkdir checkpoints
cd checkpoints
wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-car/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth
cd ..
python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth


python demo/pcd_demo.py demo/data/kitti/000008.bin configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth



python demo/pcd_demo.py demo/data/kitti/000008.bin configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-car_20200620_230238-393f000c.pth
wget 


## newer process (2023)
 export CUDA_HOME=$CONDA_PREFIX
 mkdir mmdetection3d
 cd mmdetection3d/
 git clone https://github.com/open-mmlab/mmdetection3d.git
 conda create --name openmmlab python=3.8 -y
 conda activate openmmlab
 conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
 pip install --upgrade pip
 pip install -U openmim
 mim install mmengine
 mim install 'mmcv>=2.0.0rc4'
 mim install 'mmdet>=3.0.0'
 mim install "mmdet3d>=1.1.0"


sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt-get update
(wsl --shutdown and restart)
glxinfo | grep ":" 


## tensorflow (2023)

conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
