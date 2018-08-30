clear all;
clc;
% calculate localization accuracy
% ODO_poses: poses from KITTI odometry
% GT_poses: poses calculated from velodyne ICP. Use this as a ground truth.
% EST_poses: poses estimated by depth localization.
N = 100000;
%% load poses
% fid = fopen('./kitti00_rev1/ODO_poses.txt', 'r');
% odo_poses = [];
% for iter=1:N
%     oneline = fgetl(fid);
%     if(oneline>0)
%         t_line=sscanf(oneline, '%f %f %f %f\n');
%     else
%         break;
%     end
%     odo_poses = [odo_poses;t_line'];
% end
% fclose(fid);

fid = fopen('./kitti02_slam/GT_poses.txt', 'r');
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

fid = fopen('./kitti02_slam/EST_poses.txt', 'r');
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
% N = 300;
% error_odo = [];
% error_est = [];
rot_odo = [];
rot_est = [];
tran_odo = [];
tran_est = [];
for iter=1:N
    % error matrices
    tmp_gt = gt_poses((iter-1)*3+1:iter*3,:);
    tmp_gt = [tmp_gt;0,0,0,1];
%     tmp_odo = odo_poses((iter-1)*3+1:iter*3,:);
%     tmp_odo = [tmp_odo;0,0,0,1];
    tmp_est = est_poses((iter-1)*3+1:iter*3,:);
    tmp_est = [tmp_est;0,0,0,1];
%     error_odo = pinv(tmp_gt)*tmp_odo;%[error_odo;pinv(tmp_gt)*tmp_odo];
    error_est = pinv(tmp_gt)*tmp_est;%[error_est;pinv(tmp_gt)*tmp_est];
    
    %rotational error
%     rot_odo=[rot_odo;trace(error_odo(1:3,1:3))];
    rot_est=[rot_est;trace(error_est(1:3,1:3))];
%     rot_odo=[rot_odo;(rotm2eul(error_odo(1:3,1:3)))];
%     rot_est=[rot_est;(rotm2eul(error_est(1:3,1:3)))];
    %translational error 
%     tran_odo=[tran_odo;error_odo(1:3,4)'];
    tran_est=[tran_est;error_est(1:3,4)'];
end

%% plot results
% rot_error_odo = zeros(N,1);
rot_error_est = zeros(N,1);
% tran_error_odo = zeros(N,1);
tran_error_est = zeros(N,1);
for iter=1:N
%     rot_error_odo(iter) = rad2deg(real(acos((rot_odo(iter,:)-1)/2)));
    rot_error_est(iter) = rad2deg(real(acos((rot_est(iter,:)-1)/2)));
%     tran_error_odo(iter) = sqrt(tran_odo(iter,:)*tran_odo(iter,:)');
    tran_error_est(iter) = sqrt(tran_est(iter,:)*tran_est(iter,:)');
end

maxnum = iter; 
t=1:maxnum;
t=t*100/maxnum;

hFig1 = figure(1);
plot(t,rot_error_est,'Color',[0.7,0,0]); hold on;
plot(t,repmat(mean(rot_error_est),1,numel(t)),'--','Color',[0,0,0],'LineWidth',1); hold on;
x1 = 10;
y1 = 4;
% txt1 = strcat('Average: ', num2str(mean(rot_error_est)), '\pm', num2str(std(rot_error_est)));
txt1 = strcat(num2str(mean(rot_error_est)), '\pm', num2str(std(rot_error_est)));
text(x1,y1,txt1,'FontSize',15); hold on;
% xlabel('index');
ylabel('rotational error (\circ)','FontSize',12);
axis([0 100 0 5]);
set(hFig1, 'Position', [100 900 800 300]);


hFig2 = figure(2);
plot(t,abs(tran_est(:,1)),'r'); hold on;
plot(t,abs(tran_est(:,2)),'g'); hold on;
plot(t,abs(tran_est(:,3)),'b'); hold on;
% xlabel('index');
ylabel('translational error (m)','FontSize',12);
legend('lateral','vertical','longitudinal');
axis([0 100 0 10]);
set(hFig2, 'Position', [100 350 800 300]);

hFig3 = figure(3);
plot(t,tran_error_est,'Color',[0,0,0.7]); hold on;
plot(t,repmat(mean(tran_error_est),1,numel(t)),'--','Color',[0,0,0],'LineWidth',1); hold on;
x2 = 10;
y2 = 0.8;
% txt2 = strcat('Average: ', num2str(mean(tran_error_est)), '\pm', num2str(std(tran_error_est)));
txt2 = strcat(num2str(mean(tran_error_est)), '\pm', num2str(std(tran_error_est)));
text(x2,y2,txt2,'FontSize',15); hold on;
% xlabel('index');
ylabel('translational error (m)','FontSize',12);
axis([0 100 0 10]);
set(hFig3, 'Position', [100 0 800 300]);

mean(tran_error_est)
std(tran_error_est)
mean(rot_error_est)
std(rot_error_est)


% print(hFig1,'../figures/KITTI00_rotation','-dpdf');
% print(hFig3,'../figures/KITTI00_translation','-dpdf');
% print(hFig2,'../figures/KITTI00_transxyz','-dpdf');