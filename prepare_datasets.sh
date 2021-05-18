set -ex
unzip

wget https://dataset.ait.ethz.ch/downloads/IJNQ4hZGrB/checkpoints.zip
unzip checkpoints.zip
rm checkpoints.zip

mkdir -p datasets/test
wget https://dataset.ait.ethz.ch/downloads/IJNQ4hZGrB/datasets/nocs_det.zip  -P ./datasets/test/
unzip ./datasets/test/nocs_det.zip -d ./datasets/test/
rm ./datasets/test/nocs_det.zip
 
wget http://download.cs.stanford.edu/orion/nocs/real_test.zip  -P ./datasets/test/
unzip ./datasets/test/real_test.zip -d ./datasets/test/
rm ./datasets/test/real_test.zip

mkdir -p datasets/train
wget https://dataset.ait.ethz.ch/downloads/IJNQ4hZGrB/datasets/train/bottle.hdf5  -P ./datasets/train/
wget https://dataset.ait.ethz.ch/downloads/IJNQ4hZGrB/datasets/train/bowl.hdf5  -P ./datasets/train/
wget https://dataset.ait.ethz.ch/downloads/IJNQ4hZGrB/datasets/train/camera.hdf5  -P ./datasets/train/
wget https://dataset.ait.ethz.ch/downloads/IJNQ4hZGrB/datasets/train/can.hdf5  -P ./datasets/train/
wget https://dataset.ait.ethz.ch/downloads/IJNQ4hZGrB/datasets/train/laptop.hdf5  -P ./datasets/train/
wget https://dataset.ait.ethz.ch/downloads/IJNQ4hZGrB/datasets/train/mug.hdf5  -P ./datasets/train/

