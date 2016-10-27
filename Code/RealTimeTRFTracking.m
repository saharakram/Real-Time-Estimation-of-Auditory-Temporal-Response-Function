%Tracking Temporal Response Function (TRF) of the brain using weighted gaussian Kernels

function [theta_hat,b_u,c] = RealTimeTRFTracking( varargin )

% Real-time TRF estimation with confidence interval

%Outputs:

% 1) theta_hat : Biased estimation of the TRF via SPARLS (the first and second half of the
% array represents TRFs for the first and second speakers, respectively.)

% 2) b_u : de-biased estimate of the TRF.

% 3) c : estimated confidence inervals


% Sahar Akram, sahar.akram@gmail.com
% (Akram et. al., Dynamic Estimation of the Auditory Temporal Response Function from MEG in
% Competing-Speaker Environments, Transactions on Biomedical Engineering, 2016)


Sim = 1; % Generate simulated data

Trial_Dur = 20; % Duration of a trial in sec

Fs = 200; % Sampling frequency

SNR = 10; % Signal to noise ratio for the simulated neural data


%TRF params

M = 100; % TRF length (for 200Hz this is half a second of TRF)

TRF = 1; % 0: only attended, 1: attended and unattended


%EM params for SPARLS

lambda = 0.999; % state transition scale

eta = .07; % Regularization parameter

iter_EM = 2; % Total number of EM iterations in SPARLs

sigma = 1; % Noise variance

alpha = .1; % Step-size parameter


%Conf param

ConfCo = 1.96; % 95 conf. interval


%Gaussian Kernel params

nlevel = 7; % for 5 ms resolution


%% Simulation

if Sim
    
    Frames = gaussian_basis(200,5:5:1000,0); % Generate Gaussian Kernels
    
    % Simulated weights for the 1st half of the trial
    Weights1=zeros(size(Frames,2),1);
    Weights1(10)=.1; Weights1(20)=-.4; Weights1(110)=.1; Weights1(120)=-.1;
    
    % Simulated weights for the 2nd half of the trial
    Weights2=zeros(size(Frames,2),1);
    Weights2(10)=.1; Weights2(20)=-.1; Weights2(110)=.1; Weights2(120)=-.4;
    
    M = size(Frames,1);
    nframes = size(Frames,2);
    n = Trial_Dur*Fs;
    
    %generate the covariates
    
    load('Enva1_200.mat')
    Env=Env(1:n);
    Env1=Env;
    Env_m = 20*log10(abs(Env));
    Env_m = Env_m - mean(Env_m);
    
    load('Envb1_200.mat')
    Env=Env(1:n);
    Env2=Env;
    Env_f = 20*log10(abs(Env));
    Env_f = Env_f - mean(Env_f);
    
    % Envelope Normalization
    
    Env_m=1/(sqrt(M*var(Env_m)))*Env_m;
    Env_f=1/(sqrt(M*var(Env_f)))*Env_f;
    
    %Generate the Covariates
    
    for k=1:n-M+1
        X(k,1:M/2) = (Env_m(k+M/2-1:-1:k))'; % covariates
        X(k,M/2+1:M) = (Env_f(k+M/2-1:-1:k))'; % covariates
    end;
    
    
    % Simulate MEG Signals (One attention switch in the middle of the trial)
    
    MEG(1:floor(n/2-M/2),1) = X(1:floor(n/2-M/2),:)*Frames*Weights1;
    MEG(floor(n/2-M/2)+1:n-M+1,1) = X(floor(n/2-M/2)+1:n-M+1,:)*Frames*Weights2;
    
    % Adjust the SNR
    
    var_Sig = var(MEG);
    var_Noise = var_Sig/10^(SNR/10);
    sigma = sqrt(var_Noise);
    MEG = MEG+sqrt(var_Noise)*randn(length(MEG),1);
    MEG = MEG - mean(MEG);
    
    % Estimate TRF (theta) using Lasso
    
    %[theta1 MSE NMSE] = lasso_m(X*Frames, MEG, nframes,eta,1,30);
    %theta_lasso = Frames*theta1';
else
    
    % Use experimental data
    
end

X=X*Frames;
[n,p]=size(X);


for i=1:p
    Y_temp{i}=X(:,i);
    X_temp{i}=[X(:,1:i-1) X(:,i+1:end)];
end

%% Initialization

D=diag(n-1,n-1);

for i=1:n
    D(i,i)=lambda^(n-i);
end

%Parameter adjustments

alpha = sigma/80;
sigma2 = 1/sqrt(2);
alpha2 = sigma2/85;

%Initialize B and u

B1{1} = eye(p) - (alpha/sigma)^2*X(1,:)'*X(1,:);
u1{1} =(alpha/sigma)^2*X(1,:)*MEG(1)';
theta_hat{1} = zeros(p,1);


for i=1:p
    B2{1,i}= eye(p-1) - (alpha2/sigma2)^2*X_temp{i}(1,:)'*X_temp{i}(1,:);
    u2{1,i}=(alpha2/sigma2)^2*X_temp{i}(1,:)*Y_temp{i}(1)';
    gamma_hat{1,i}=zeros(p-1,1);
end


D_sq= diag(sqrt(diag(D)));
Sigma_0{1}=X(1,:)'*X(1,:);
Sigma_00{1}=X(1,:)'*X(1,:);
temp1{1}=X(1,:)'*MEG(1);

for i=1:p
    Sigma_1{1,i}= X(1,i)'*X(1,[1:i-1 i+1:end]);
end


for t=2:n
    
    B1{2}=lambda*B1{1}-(alpha/sigma)^2*X(t,:)'*X(t,:)+(1-lambda)*eye(p);
    u1{2}=lambda*u1{1}+(alpha/sigma)^2*MEG(t)'*X(t,:);
    
    %Run LCEM
    
    r(:,1)= B1{2}*theta_hat{t-1}+u1{2}';
    I_p{1}=find(r(:,1)>eta*alpha^2);
    I_n{1}=find(r(:,1)<-eta*alpha^2);
    I_0{1}= find(~ismember((1:p),[I_p{1};I_n{1}]));
    ww=[];
    
    % EM iterations
    
    for l = 2:iter_EM
        
        ww(I_p{l-1},l)= r(I_p{l-1},l-1) - eta*alpha*2;
        ww(I_n{l-1},l)= r(I_n{l-1},l-1) + eta*alpha*2;
        ww(I_0{l-1},l)= 0;
        
        r(:,l)=B1{2}*ww(:,l)+u1{2}';
        I_p{l}=find(r(:,l)>eta*alpha^2);
        I_n{l}=find(r(:,l)<-eta*alpha^2);
        I_0{l}= find(~ismember((1:p),[I_p{l};I_n{l}]));
        
    end
    
    
    theta_hat{t}(I_p{l},1)= r(I_p{l},l)-eta*alpha*2;
    theta_hat{t}(I_n{l},1)= r(I_n{l},l)+eta*alpha*2;
    theta_hat{t}(I_0{l},1)= 0;
    
    
    %update Sigma
    
    Sigma_0{2}=lambda* Sigma_0{1}+X(t,:)'*X(t,:);
    Sigma_00{2}=lambda^2* Sigma_00{1}+X(t,:)'*X(t,:);
    temp1{2}=lambda*temp1{1}+X(t,:)'* MEG(t);
    
    tic
    
    %% Node-wise regression with SPARLS
    
    for i=1:p
        
        
        B2{2,i}=lambda*B2{1,i}-(alpha2/sigma2)^2*X_temp{i}(t,:)'*X_temp{i}(t,:)+(1-lambda)*eye(p-1);
        u2{2,i}=lambda*u2{1,i}+(alpha2/sigma2)^2*Y_temp{i}(t)'*X_temp{i}(t,:);
        
        %Run LCEM
        
        rr(:,1)= B2{2}*gamma_hat{1,i}+u2{2}';
        I_p{1}=find(rr(:,1)>eta*alpha2^2);
        I_n{1}=find(rr(:,1)<-eta*alpha2^2);
        I_0{1}= find(~ismember((1:p-1),[I_p{1};I_n{1}]));
        
        ww=[];
        
        for l=2:iter_EM
            
            ww(I_p{l-1},l)= rr(I_p{l-1},l-1)-eta*alpha2*2;
            ww(I_n{l-1},l)= rr(I_n{l-1},l-1)+eta*alpha2*2;
            ww(I_0{l-1},l)= 0;
            
            
            rr(:,l)=B2{2,i}*ww(:,l)+u2{2,i}';
            I_p{l}=find(rr(:,l)>eta*alpha2^2);
            I_n{l}=find(rr(:,l)<-eta*alpha2^2);
            I_0{l}= find(~ismember((1:p-1),[I_p{l};I_n{l}]));
            
        end
        
        gamma_hat{2,i}(I_p{l},1)= rr(I_p{l},l)-eta*alpha2*2;
        gamma_hat{2,i}(I_n{l},1)= rr(I_n{l},l)+eta*alpha2*2;
        gamma_hat{2,i}(I_0{l},1)= 0;
        
        %Update Sigma_1
        
        Sigma_1{2,i} = lambda*Sigma_1{1,i} + X(t,i)'*X(t,[1:i-1 i+1:end]);
        tau2(i) = Sigma_0{2}(i,i) - Sigma_1{2,i}*gamma_hat{2,i};
        
    end
    
    C=[];
    
    for i=1:p
        
        if i==1
            C(:,1)=[1; -gamma_hat{2,1}];
        elseif i==p
            C(:, p)=[-gamma_hat{2,p}; 1];
        else
            C(:,i)=[-gamma_hat{2,i}(1:i-1); 1; -gamma_hat{2,i}(i:p-1)];
        end
        
    end
    
    Theta=(diag(1./tau2))*C;
    
    %% Estimating b_u(unbiased estimate of the TRF) and c(confidence intervals)
    
    b_u{t} = theta_hat{t}+Theta*(temp1{2}-Sigma_0{2}*theta_hat{t});
    
    K=Theta*Sigma_00{2}*Theta';
    K1=Frames*Theta*Sigma_00{2}*Theta'*Frames';
    
    c{t}=ConfCo*sigma*sqrt(diag(K));
    C_U{t}=ConfCo*sigma*sqrt(diag(K1));
    
    tau{t}=Frames*theta_hat{t};
    B_U{t}=Frames*b_u{t};
    
    toc
    
    %% ploting
    
    if mod(t,10)==0 % plot after every 10 analysis steps
        
        % Simulated TRF
        subplot(2,1,1)
        if Sim
            if t < n/2
                w1 = Frames*Weights1;
                plot((1:M)*5,w1(1:M),'r')
            else
                w1 = Frames*Weights2;
                plot((1:M)*5,w1(1:M),'r')
            end
        end
        
        % Estimated TRF
        subplot(2,1,2)
        errorbar((1:M)*5,tau{t}(1:M),-B_U{t}(1:M)+tau{t}(1:M)+C_U{t}(1:M),B_U{t}(1:M)-tau{t}(1:M)+C_U{t}(1:M),'.')
        xlim([1*5 M*5])
        if Sim
            ylim([-.3 .3])
        else
            ylim([-.01 .01])
        end
        t
        pause
    end
    
    %% update B1, U1, Sigma_0,temp1,
    
    B1{1}=B1{2};
    u1{1}=u1{2};
    Sigma_0{1}=Sigma_0{2};
    Sigma_00{1}=Sigma_00{2};
    temp1{1}=temp1{2};
    
    B2{1}=B2{2};
    u2{1}=u2{2};
    Sigma_1{1}=Sigma_1{2};
    
end

m



