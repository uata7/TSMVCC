function unew = TSMVCC_grad(uall, normall, T,Tnorms,r,tau)
% T      : the common distance matrix 
% uall   : the view distance matrix

n = size(T, 1); 
c= size(T, 2);
numviews=length(uall);
uuualls = vertcat(uall{:});
nnnorms = vertcat(normall{:});
Unornv=uuualls ./nnnorms; 
Tnorn=T./Tnorms; 
S = Unornv * Tnorn'; 
eSnv=exp(S / tau);
Bs = sum(eSnv, 2);    
eSnvBs=eSnv./Bs;
so1=sum(eSnvBs.*S,2);
grad1=eSnvBs*Tnorn-so1.*Unornv;
reTnorn = repmat(Tnorn, numviews, 1); 
Siinv = S(sub2ind([n*numviews, n], (1:n*numviews)', mod((1:n*numviews)'-1, n)+1)); 
grad2=(Siinv.*Unornv-reTnorn);
gradU=(grad1+grad2)./nnnorms;
gradUij=gradU.* (uuualls.^r);
unew = mat2cell(gradUij, repmat(n, numviews, 1), c); 
end