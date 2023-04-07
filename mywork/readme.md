
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


(abhi) cs21resch15003@dgx1-1:~/cs22mds15010/mmdet/mmdetection3d/data/kitti$ ln -s /raid/cs21resch15003/cs22mds15010/dataset/KITTI/training/ training
(abhi) cs21resch15003@dgx1-1:~/cs22mds15010/mmdet/mmdetection3d/data/kitti$ ln -s /raid/cs21resch15003/cs22mds15010/dataset/KITTI/testing/ testing


python mmdet3d/utils/collect_env.py

https://mmdetection3d.readthedocs.io/en/v0.15.0/supported_tasks/lidar_det3d.html

python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
python mywork/train.py


04/08 00:55:36 - mmengine - INFO - ------------------------------
04/08 00:55:36 - mmengine - INFO - The length of the dataset: 3712
04/08 00:55:36 - mmengine - INFO - The number of instances per category in the dataset:
+----------------+--------+
| category       | number |
+----------------+--------+
| Pedestrian     | 2218   |
| Cyclist        | 846    |
| Car            | 14398  |
| Van            | 1425   |
| Truck          | 540    |
| Person_sitting | 92     |
| Tram           | 238    |
| Misc           | 469    |
+----------------+--------+
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3712/3712, 165.0 task/s, elapsed: 22s, ETA:     0s
load 14398 Car database infos
load 238 Tram database infos
load 846 Cyclist database infos
load 1425 Van database infos
load 469 Misc database infos
load 2218 Pedestrian database infos
load 540 Truck database infos
load 92 Person_sitting database infos



git config --global user.email "abhiyantaabhishek1@gmail.com"
git config --global user.name "Abhishek"

git init
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:abhiyantaabhishek/mmdetection3d.git
git push -u origin main