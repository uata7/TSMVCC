function [Y,obj_history] = TSMVCC(X,c,max_iters,tolerance,q,r,tau,h,a,beta,l)
% X      : n*dv
% V      : common cluster center matrix 
% T      : the common distance matrix 
% uall   : the view distance matrix
% max_iters  : the maximum number of iterations in Stage 2

%%  initialize 
rng(4, 'twister');    
numview = length(X); 
n = size(X{1}, 1);
X_concat = [X{:}];
[label1, ~] = litekmeans(X_concat, c, 'MaxIter', 100, 'Replicates', 10);
n1 = length(label1);
Y = full(sparse(1:n1, label1, 1, n1, c));
XWc=cell(numview, 1);
M = cell(numview, 1);
YTT=(Y.^r)'; 
YTTT=sum(YTT,2);
AA=cell(max_iters,1);
momenM = cell(numview, 1); 
  
for v = 1:numview    
M{v}= YTT*X{v} ./ YTTT;
momenM{v} = zeros(size(M{v}));
end

alpha = ones(numview, 1) / numview;  
alphaq=alpha.^q;

H = rand(n, h);
momenH = zeros(size(H));
H = mapminmax(H', 0, 1);  
H=H'; 

%%  stage 1  
for iter = 1:5
  %% optimize W^i
for i=1:numview
                iw = X{i}'*H;
                [ip,~,iq] = svd(iw,'econ');
                W{i} = ip*iq'; 
                XWc{i}=X{i}*W{i};
end  
sumXW = sum(cat(3, XWc{:}), 3); 
  %% optimize H
H=sumXW/numview;
end

%%  Stage 2    initialize 
V= YTT*H  ./ YTTT; 


uall = cell(numview, 1);
normall = cell(numview, 1); 
momenV=zeros(size(V)); 
d = zeros(numview, 1);
for  v = 1:numview
     dtt = pdist2(X{v}, M{v},'euclidean')+eps;
     uall{v} = dtt.^(2/(1-r)); 
     normall{v} = sqrt(sum(uall{v}.^2, 2));
     d(v)=size(M{v}, 2);
end 
dttt = pdist2(H , V ,'euclidean')+eps;
T  = dttt.^(2/(1-r)); 
Tnorms=sqrt(sum(T .^2, 2)); 


obj_history = zeros(max_iters, 1); 

for iter = 1:max_iters 

%% optimize H
tnew = TSMVCC_grad2(alpha,q,uall,normall,T,Tnorms,r,tau); 
xsm=(2)/((1-r)*tau*n);  
xsH=xsm*(h^l); 
dlc_dH=xsH* (sum(tnew,2).*H-tnew*V);
dl_dH=dlc_dH; 
momenH = beta * momenH + dl_dH;
H = H - a *  momenH;
dttt = pdist2(H , V ,'euclidean')+eps;
T  = dttt.^(2/(1-r));
Tnorms=sqrt(sum(T .^2, 2)); 

%% optimize M^v
unew = TSMVCC_grad(uall, normall, T,Tnorms,r,tau);
 for v0 = 1:numview  
     unewc1=sum(unew{v0},1)';
     dlcc_dm= ((unewc1).*M{v0}-(unew{v0}'*X{v0}))*xsm*(d(v0)^l)*alphaq(v0); 
     momenM{v0} = beta * momenM{v0} + dlcc_dm;
     M{v0} = M{v0} - a * momenM{v0};
     uall{v0}=(pdist2(X{v0}, M{v0},'euclidean')+eps).^(2/(1-r));
     normall{v0} = sqrt(sum(uall{v0}.^2, 2)); 
 end

%% optimize V
tnew = TSMVCC_grad2(alpha,q,uall,normall,T,Tnorms,r,tau); 
sstnew=sum(tnew,1)';
dl_dV=xsH*(sstnew.*V -tnew'*H);
momenV = beta * momenV + dl_dV;
V = V - a *  momenV;

%% optimize alpha 
dttt = pdist2(H , V ,'euclidean')+eps;
T  = dttt.^(2/(1-r));
Tnorms=sqrt(sum(T .^2, 2)); 
uuualls = vertcat(uall{:});
nnnorms = vertcat(normall{:});
Unornv=uuualls ./nnnorms; 
Tnorn=T./Tnorms; 
S = Unornv * Tnorn'; 
eSnv=exp(S / tau);
Bs = sum(eSnv, 2);  
Siinv = S(sub2ind([n*numview, n], (1:n*numview)', mod((1:n*numview)'-1, n)+1));
ivnv=log(Bs)-Siinv/tau;
Avvalpha = sum(reshape(ivnv, n, numview), 1)'; 
Aq=Avvalpha.^(1/(1-q));
alpha=Aq/(sum(Aq));
alphaq=alpha.^q;

%%
totalloss=sum(alphaq.*Avvalpha)*(1/n);
obj_history(iter) = totalloss;

    if iter > 1 && abs(obj_history(iter) - obj_history(iter-1)) < tolerance
        break;
    end
end
  Y=T./sum(T,2); 
end