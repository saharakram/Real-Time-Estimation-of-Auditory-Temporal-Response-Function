function WW = gaussian_basis(nlevel,range,all)

x=range;
W=[];
if all==1
for i=0:1:nlevel
    
    means=[x(end)/2^(i+1):x(end)/2^i:x(end)*(1-1/2^(i+1))];
    
    stds = zeros(1,2^i)+100/2^i;
    
    for count=1:2^i
    
        W = cat(1,W,exp(-(x - means(count)).^2/(2*stds(count)^2)));
              
    end
end

else
    means=[x(end)/nlevel:x(end)/nlevel:x(end)*(1-1/nlevel)];
    
    stds = zeros(1,nlevel-1)+8.5; 
    
    for count=1:nlevel-1
    
        W = cat(1,W,exp(-(x - means(count)).^2/(2*stds(count)^2)));
              
    end
end

W=W';
WW=bsxfun(@rdivide,W,max(W));



% figure;
% 
% for i=0:1:nlevel
%     c=mod([1 0 1]+[.2 .2 .2],1);
%     plot(x,W(2^i:2^(i+1)-1,:),'Color',c)
%     hold on
% end
