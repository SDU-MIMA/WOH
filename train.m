function [MAP_result] = train(X, Y, Z, param,L,XTest,LTest,anchor)

%% set the parameters
nbits = param.nbits;

mu = param.mu;
beta = param.beta;
alpha = param.alpha;
theta = param.theta;
chunk = param.chunk;
MAP_result=zeros(1,8);  % MIRFlickr, 8 chunks

%% get the dimensions of features
n = size(X,1);
dX = size(anchor,1);
dY = size(Y,2);

%% initialization
B = sign(randn(n, nbits));
W = randn(nbits, dY);
P = randn(dX, nbits);


%% iterative optimization
for round = 1:(n/chunk)
    fprintf('chunk %d: training starts. \n',round)
    e_id = (round-1)*chunk+1;
    n_id = min(round*chunk,n);
    if round == 8 && strcmp(param.datasets,'MIRFlickr')
        n_id = n;
    end
    XTrain=Kernelize(X(e_id:n_id,:),anchor);
    if round == 1
        for iter = 1:param.iter
            C1 = B(e_id:n_id,:)'*B(e_id:n_id,:);
            C2 = B(e_id:n_id,:)'*XTrain;
            C3 = XTrain'*XTrain;
            C4 = XTrain'*B(e_id:n_id,:);
            C5 = B(e_id:n_id,:)'*Z(e_id:n_id,:);
      
            % update U
            U = pinv(C1+(alpha/beta)*eye(nbits))*(C2);
            
            % update P
            P = pinv(C3+(alpha/mu)*eye(dX))*(C4);

            % update V
            V = pinv(C1+(alpha/theta)*eye(nbits))*(C5);

            % update W
            k = 1./(sqrt(sum((Y(e_id:n_id,:)-B(e_id:n_id,:)*W).^2,2)));
            K = diag(k);
            D1 = B(e_id:n_id,:)'*K*B(e_id:n_id,:);
            D2 = B(e_id:n_id,:)'*K*Y(e_id:n_id,:);
            W = pinv(D1+alpha*eye(nbits))*(D2);
            
            % update B
            Q = K*Y(e_id:n_id,:)*W'+beta*(XTrain*U')+mu*(XTrain*P)+theta*(Z(e_id:n_id,:)*V');
            for j = 1:3
                for i = 1:nbits
                    bit = 1:nbits;
                    bit(i) = [];
                    B(e_id:n_id,i) = (sign(Q(:,i)'-W(i,:)*W(bit,:)'*B(e_id:n_id,bit)'*K-beta*U(i,:)*U(bit,:)'*B(e_id:n_id,bit)'-theta*V(i,:)*V(bit,:)'*B(e_id:n_id,bit)'))';
                end
            end
            loss = sum(sqrt(sum(Y(e_id:n_id,:)-B(e_id:n_id,:)*W.^2,2))) + beta*norm(XTrain-B(e_id:n_id,:)*U,'fro') + mu*norm(B(e_id:n_id,:)-XTrain*P,'fro') + theta*norm(Z(e_id:n_id,:)-B(e_id:n_id,:)*V,'fro');


        end
    else
        CC1 = C1; CC2 = C2;  CC3 = C3; CC4 = C4;  CC5 = C5; DD1 = D1;  DD2 = D2;
        for iter = 1:param.iter
            C1 = CC1+B(e_id:n_id,:)'*B(e_id:n_id,:);
            C2 = CC2+B(e_id:n_id,:)'*XTrain;
            C3 = CC3+XTrain'*XTrain;
            C4 = CC4+XTrain'*B(e_id:n_id,:);
            C5 = CC5+B(e_id:n_id,:)'*Z(e_id:n_id,:);

            % update U
            U = pinv(C1+(alpha/beta)*eye(nbits))*(C2);
     
            % update P
            P = pinv(C3+(alpha/mu)*eye(dX))*(C4);

            % update V
            V = pinv(C1+(alpha/theta)*eye(nbits))*(C5);

            % update W
            k = 1./(sqrt(sum((Y(e_id:n_id,:)-B(e_id:n_id,:)*W).^2,2)));
            K = diag(k);
            W = pinv(DD1+B(e_id:n_id,:)'*K*B(e_id:n_id,:)+alpha*eye(nbits))*(DD2+B(e_id:n_id,:)'*K*Y(e_id:n_id,:));
            
            D1 = DD1+B(e_id:n_id,:)'*K*B(e_id:n_id,:);
            D2 = DD2+B(e_id:n_id,:)'*K*Y(e_id:n_id,:);

            % update B
            Q = K*Y(e_id:n_id,:)*W'+beta*(XTrain*U')+mu*(XTrain*P)+theta*(Z(e_id:n_id,:)*V');
            for j = 1:3
                for i = 1:nbits
                    bit = 1:nbits;
                    bit(i) = [];
                    B(e_id:n_id,i) = (sign(Q(:,i)'-W(i,:)*W(bit,:)'*B(e_id:n_id,bit)'*K-beta*U(i,:)*U(bit,:)'*B(e_id:n_id,bit)'-theta*V(i,:)*V(bit,:)'*B(e_id:n_id,bit)'))';
                end
            end
        end
    end
    
    fprintf('       : training ends, evaluation begins. \n')
    XKTest=Kernelize(XTest,anchor);
    BxTest = compactbit(XKTest*P >= 0);
    BxTrain = compactbit(B(1:n_id,:) >= 0);
    DHamm = hammingDist(BxTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    MAP  = mAP(orderH', L(1:n_id,:), LTest);
    fprintf('       : evaluation ends, MAP is %f\n',MAP);
    MAP_result(1,round)=MAP;

end
