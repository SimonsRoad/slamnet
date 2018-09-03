%generating training txt file
% image_left, image_right, image_left_next, image_right, 
% left_focal_length, left_c0, left_c1, 
% right_focal_length, right_c0, right_c1
% base_line, width, height 

data_num = [4500,1100,4600, 800, 200, 2700, 1100, 1100, 4000];
image_dims = [376, 376, 376, 375, 370, 370, 370 ,370, 370;
              1241, 1241, 1241, 1242, 1226, 1226, 1226, 1226, 1226];

left_focal_length = zeros(1,numel(data_num));
right_focal_length = zeros(1,numel(data_num));
left_c0 = zeros(1,numel(data_num));
left_c1 = zeros(1,numel(data_num));
right_c0 = zeros(1,numel(data_num));
right_c1 = zeros(1,numel(data_num));
base_line = zeros(1,numel(data_num));
base_str = '/media/youngji/storagedevice/naver_data/kitti_odometry/dataset/';
for dnum=1:numel(data_num)
    %read camera param
    num_str = sprintf('sequences/%02d/calib.txt',dnum-1);
    read_str = strcat(base_str,num_str);
    read_fid = fopen(read_str,'r');
    lines = fscanf(read_fid, '%s %f %f %f %f %f %f %f %f %f %f %f %f\n',[15,5]);
    left_focal_length(dnum) = lines(4,3);
    right_focal_length(dnum) = lines(4,4);
    left_c0(dnum) = lines(6,3);
    left_c1(dnum) = lines(10,3);
    right_c0(dnum) = lines(6,4);
    right_c1(dnum) = lines(10,4);
    base_line(dnum) = -lines(7,2)/lines(4,2);
    fclose(read_fid);
end

save_fid = fopen('../utils/filenames/kitti_train_files.txt', 'wt');
for dnum=1:numel(data_num)
    % read poses
    num_str = sprintf('poses/%02d.txt',dnum-1);
    read_str = strcat(base_str,num_str);
    read_fid = fopen(read_str,'r');
    lines = fscanf(read_fid, '%f %f %f %f %f %f %f %f %f %f %f %f\n',[12,data_num(dnum)+1]);
    for iter=1:data_num(dnum)
        % compute pose
        cur_data = lines(:,iter+1);
        tran_mat = reshape(cur_data,[4,3])';
        eul = rotm2eul(tran_mat(1:3,1:3));
        % write data
        fprintf(save_fid, '%d ', dnum-1);
        fprintf(save_fid, 'sequences/%02d/image_2/%06d.png ', dnum-1, iter-1);
        fprintf(save_fid, 'sequences/%02d/image_2/%06d.png ', dnum-1, iter);
        fprintf(save_fid, '%f %f %f ', left_focal_length(dnum), left_c0(dnum), left_c1(dnum));
        fprintf(save_fid, '%f %f %f ', right_focal_length(dnum), right_c0(dnum), right_c1(dnum));
        fprintf(save_fid, '%f %f %f ', base_line(dnum), image_dims(1,dnum), image_dims(2,dnum));
        fprintf(save_fid, '%f %f %f ', eul(1), eul(2), eul(3));
        fprintf(save_fid, '%f %f %f ', tran_mat(1,4), tran_mat(2,4), tran_mat(3,4));
        fprintf(save_fid, '\n');
    end
end
fclose(save_fid);

