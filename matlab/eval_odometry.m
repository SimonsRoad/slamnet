%% evaluate visual odometry
clear all;
clc;
N = 100000;

%% read results
fid = fopen('./kitti10/GT_poses.txt', 'r');
gt_poses = [];
for iter=1:N
    oneline = fgetl(fid);
    if(oneline>0)
        t_line=sscanf(oneline, '%f %f %f %f\n');
    else
        break;
    end
    gt_poses = [gt_poses;t_line'];
end
fclose(fid);

fid = fopen('./kitti10/VO_poses.txt', 'r');
est_poses = [];
for iter=1:N
    oneline = fgetl(fid);
    if(oneline>0)
        t_line=sscanf(oneline, '%f %f %f %f\n');
    else
        break;
    end
    est_poses = [est_poses;t_line'];
end
fclose(fid);

%% calculate error
N = size(gt_poses,1)/3;
rot_odo = [];
rot_est = [];
tran_odo = [];
tran_est = [];
len = [];
tran_gt = [];
for iter=1:N
    % error matrices
    tmp_gt = gt_poses((iter-1)*3+1:iter*3,:);
    tmp_gt = [tmp_gt;0,0,0,1];
    tran_gt = [tran_gt;tmp_gt(1:3,4)'];

    tmp_est = est_poses((iter-1)*3+1:iter*3,:);
    tmp_est = [tmp_est;0,0,0,1];
    
    error_est = pinv(tmp_gt)*tmp_est;
    
    %rotational error
    rot_est=[rot_est;trace(error_est(1:3,1:3))];
    %translational error 
    tran_est=[tran_est;error_est(1:3,4)'];

    if(iter>1)
        delta = pinv(prev_gt)*tmp_gt;
        len_prev = len_prev + delta(1:3,4)'*delta(1:3,4);
        len = [len,len_prev];
    else
        len_prev = 0;
        len = [len,0];
    end
    prev_gt = tmp_gt;
end

%% plot results
rot_error_est = zeros(8,1);
tran_error_est = zeros(8,1);
lengths = [100.0,200.0,300.0,400.0,500.0,600.0,700.0,800.0];
count = 1;
for iter=1:N
    if len(iter)>lengths(count)
        rot_error_est(count) = rad2deg(real(acos((rot_est(iter,:)-1)/2)))/len(iter);
        tran_error_est(count) = sqrt(tran_est(iter,:)*tran_est(iter,:)')/len(iter);
        count = count+1;
    end
    if count>8
        break;
    end
end

hFig1 = figure(1);
plot(lengths,rot_error_est,'Color',[0.7,0,0]); hold on;
plot(lengths,tran_error_est,'Color',[0,0.7,0]); hold on;

mean(tran_error_est)*100
mean(rot_error_est)*100

ylabel('rotational error (\circ)','FontSize',12);

