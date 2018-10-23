0. Prepare dataset
run generate_train.m with matlab 

1. Activate tensorflow
source ~/tensorflow/bin/activate 

2. Training
python deepslam_main.py --mode train --model_name deepslam_model --data_path /media/youngji/storagedevice/naver_data/kitti_odometry/dataset/ --filenames_file ./utils/filenames/kitti_train_files.txt --log_directory ./tmp/ --vo_checkpoint_path ./tmp/updeepvo_model/model-76500


3. visualization
tensorboard --logdir=./tmp/my_model
 
4. Testing
python updeepvo_test.py --checkpoint_path ./tmp/my_model/model-10000
python deepslam_test.py --checkpoint_path ./tmp/deepslam_model/model-93599


python deepslam_main.py --mode train --model_name deepslam_model --data_path /media/youngji/storagedevice/naver_data/kitti_odometry/dataset/ --filenames_file ./utils/filenames/kitti_train_files.txt --log_directory ./tmp/ --vo_checkpoint_path ./tmp/updeepvo_model/model-76500 --checkpoint_path ./tmp/deepslam_model3/model-32159

