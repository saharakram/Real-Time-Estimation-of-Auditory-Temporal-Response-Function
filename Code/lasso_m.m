function [theta MSE NMSE] = lasso_me(X, Y,M,alpha,win,iter)

Y = smooth(Y,win);


theta(1:M)=0;

for l=1:iter
    WW = 0;
    for k=1:M
    WW(k,k) = (1/sqrt(theta(k)^2 + 0.000000000000000000001));
    end;
    W = X'*X + alpha*WW;
    for j=1:20 % iterations of Newton's method
    err = inv(W)*(X'*(Y - X*theta') - alpha*WW*theta');
    theta = theta + err';
    end
end;


MSE=var(Y - X*theta');
NMSE = var(Y - X*theta')/var(Y);