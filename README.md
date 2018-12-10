0. Prepare dataset
run generate_train.m with matlab 

1. Activate tensorflow
source ~/tensorflow/bin/activate 

2. Training
[mapping]
python deepslam_main.py --mode train --train_mode mapping --model_name mapping_model --data_path /media/youngji/storagedevice/naver_data/kitti_odometry/dataset/ --filenames_file ./utils/filenames/kitti_train_files.txt --log_directory ./tmp/ --prev_checkpoint_path ./tmp/pose_model/model-76500

[localization]
python deepslam_main.py --mode train --train_mode localization --model_name localization_model --data_path /media/youngji/storagedevice/naver_data/kitti_odometry/dataset/ --filenames_file ./utils/filenames/kitti_train_files.txt --log_directory ./tmp/ --prev_checkpoint_path ./tmp/mapping_model/model-76500

[slam]
python deepslam_main.py --mode train --train_mode slam --model_name slam_model --data_path /media/youngji/storagedevice/naver_data/kitti_odometry/dataset/ --filenames_file ./utils/filenames/kitti_train_files2.txt --log_directory ./tmp/ --prev_checkpoint_path ./tmp/pose_model2/model-119700

3. visualization
tensorboard --logdir=./tmp/my_model
 
4. Testing
python updeepvo_test.py --checkpoint_path ./tmp/my_model/model-10000
python deepslam_test.py --checkpoint_path ./tmp/deepslam_model/model-93599

python deepslam_main.py --mode train --train_mode slam --model_name slam_model3 --data_path /media/youngji/storagedevice/naver_data/kitti_odometry/dataset/ --filenames_file ./utils/filenames/kitti_train_files.txt --log_directory ./tmp/ --checkpoint_path ./tmp/slam_model/model-93599

