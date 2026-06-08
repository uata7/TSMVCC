addpath(genpath('D:\xxx')); 

load('MSRC.mat','X','Y'); 
y=Y;
q=1.4;r=1.3;h=40;l=1.5;


load('UCI.mat','X','y');
q=1.5;r=1.2;h=40;l=1.6;


c = max(y);
nv = length(X); 
for ni = 1:nv
    X{ni} = mapminmax(X{ni}', 0, 1);  
end 
X = cellfun(@(x) x', X, 'UniformOutput', false);
max_iters = 100; tolerance = 1e-4; a=0.01;beta=0.9;tau=0.4;    
t_start = tic;
[Y,obj_history] = TSMVCC(X,c,max_iters,tolerance,q,r,tau,h,a,beta,l);
runtime = toc(t_start);
[~, label_out] = max(Y, [], 2);
result_yyyy = ClusteringMeasure(y, label_out);