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
tran_error_ate = [];
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
    
    % for ATE
    if iter+4<N
        tmp_gt = gt_poses(((iter+3)-1)*3+1:(iter+3)*3,:);
        tmp_gt = [tmp_gt;0,0,0,1];
        tmp_est = est_poses(((iter+3)-1)*3+1:(iter+3)*3,:);
        tmp_est = [tmp_est;0,0,0,1];        
        init_error = pinv(tmp_gt)*tmp_est;
        ate_errors = 0;
        for iter2=0:1:4
            cur_num = iter+iter2;
            
            tmp_gt = gt_poses((cur_num-1)*3+1:cur_num*3,:);
            tmp_gt = [tmp_gt;0,0,0,1];

            tmp_est = est_poses((cur_num-1)*3+1:cur_num*3,:);
            tmp_est = [tmp_est;0,0,0,1];
            tmp_est = tmp_est*pinv(init_error);
            
            error_est = pinv(tmp_gt)*tmp_est;
            ate_errors = ate_errors + error_est(1:3,4)'*error_est(1:3,4);
        end
        tran_error_ate = [tran_error_ate;sqrt(ate_errors/5)];
    end
end

%% compute average error
rot_error_percent = [];
tran_error_percent = [];
% tran_error_ate = [];
lengths = [100.0,200.0,300.0,400.0,500.0,600.0,700.0,800.0];
count = 1;
for iter=1:N
    if count<9
        if len(iter)>lengths(count)
            rot_error_percent = [rot_error_percent;rad2deg(real(acos((rot_est(iter,:)-1)/2)))/len(iter)];
            tran_error_percent= [tran_error_percent;sqrt(tran_est(iter,:)*tran_est(iter,:)')/len(iter)];
            count = count+1;
        end
    end


end

mean(tran_error_percent)*100
mean(rot_error_percent)*100
mean(tran_error_ate)
%% plot results

fig = figure;
set(fig, 'Position', [0, 0, 650, 600]);
gt_line = reshape(gt_poses(:,4),3,size(gt_poses,1)/3);
plot(gt_line(1,:), gt_line(3,:), 'r','LineWidth',1); hold on;
est_line = reshape(est_poses(:,4),3,size(est_poses,1)/3);
plot(est_line(1,:), est_line(3,:), 'g','LineWidth',1); hold on;

xlabel('x (m)','fontsize',12);
ylabel('y (m)','fontsize',12);
h = legend('GT','Propsed');
set(h,'FontSize',15);
set(h,'Location','northwest');
% view([0 0]);
axis equal;
% grid on;
axis tight;
% axis([-300 300 -50 500]);

% print(fig,'../figures/KITTI00','-dpdf');

