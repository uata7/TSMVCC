function tnew = TSMVCC_grad2(alpha,q,uall,normall,T,Tnorms,r,tau)
% T      : the common distance matrix 
% uall   : the view distance matrix

n = size(T, 1); 
numview=length(uall);
uuualls = vertcat(uall{:});
nnnorms = vertcat(normall{:});
Unornv=uuualls ./nnnorms; 
Tnorn=T./Tnorms; 
S = Unornv * Tnorn'; 
Stau=S / tau;
eSnv = exp(Stau);
Bs = sum(eSnv, 2);    
eSnvBs=eSnv./Bs; 
alphaq=alpha.^q;
alphaqnv = repelem(alphaq, n);  
eeiv= eSnvBs.* alphaqnv; 
suSeeiv=sum(eeiv.*S,1)'; 
x11=(eeiv')*Unornv; x22=suSeeiv.*Tnorn;
tnew3=x11-x22;
tnew11=tnew3./Tnorms;  
Uaqnvc=Unornv.* alphaqnv;
sumUaqn1=cell2mat(arrayfun(@(i) sum(Uaqnvc(i:n:numview*n,:), 1), (1:n)', 'UniformOutput', false));

Siinv1 = S(sub2ind([n*numview, n], (1:n*numview)', mod((1:n*numview)'-1, n)+1));
Siiaqnv1=Siinv1.* alphaqnv;
sumSiiaqn1 = arrayfun(@(i) sum(Siiaqnv1(i:n:numview*n)), (1:n))'; 

tnew22=(sumUaqn1-sumSiiaqn1.*Tnorn)./Tnorms;

tnew=(tnew11-tnew22).*(T.^r) ;

end