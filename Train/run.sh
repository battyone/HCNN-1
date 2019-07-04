CAFFE_VERSION=caffe_nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/software/caffe_lib:/usr/local/cuda-8.0/lib64:$HOME/software/opencv-3.4.1/install/lib:$HOME/software/cudnnv6.0/lib64:$HOME/software/${CAFFE_VERSION}/distribute/lib
CAFFE=$HOME/software/${CAFFE_VERSION}/distribute/bin/caffe


$CAFFE train --solver=HCNN_solver.prototxt --log_dir=./log


cd visual
./visualize_log.sh
