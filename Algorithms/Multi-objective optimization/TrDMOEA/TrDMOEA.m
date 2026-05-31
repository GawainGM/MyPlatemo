classdef TrDMOEA < ALGORITHM
% <2021> <multi> <real> <constrained/none> <dynamic>
% Transfer dynamic multi-objective evolutionary algorithm
% sampleN --- 30 --- Number of samples for transfer learning
% mu      --- 0.1 --- Regularization parameter in TCA
% dim     --- 20 --- Latent space dimension
% gamma   ---  1 --- Gaussian kernel parameter

% This implementation is adapted to the PlatEMO algorithm interface from
% the Tr-DMOEA code in DMOEAs-main/Algorithm/Tr-DMOEA.

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [sampleN,mu,dim,gamma] = Algorithm.ParameterSet(30,0.1,20,1);
            Algorithm.save = sign(Algorithm.save)*inf;

            %% Generate random population
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
            AllPop = [];
            LastDec = Population.decs;
            LastObj = Population.objs;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                if TrDMOEA_Changed(Problem,Population)
                    AllPop = [AllPop,Population];
                    InitDec = TrDMOEA_Response(Problem,Population,LastDec,LastObj,sampleN,mu,dim,gamma);
                    Population = Problem.Evaluation(InitDec);
                    [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
                    LastDec = Population.decs;
                    LastObj = Population.objs;
                else
                    MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                    Offspring  = OperatorGA(Problem,Population(MatingPool));
                    [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
                    LastDec = Population.decs;
                    LastObj = Population.objs;
                end
                if Problem.FE >= Problem.maxFE
                    Population = [AllPop,Population];
                    [~,rank]   = sort(Population.adds(zeros(length(Population),1)));
                    Population = Population(rank);
                end
            end
        end
    end
end

function changed = TrDMOEA_Changed(Problem,Population)
    RePop1  = Population(randperm(end,ceil(end/10)));
    RePop2  = Problem.Evaluation(RePop1.decs);
    changed = ~isequal(RePop1.objs,RePop2.objs) || ~isequal(RePop1.cons,RePop2.cons);
end

function InitDec = TrDMOEA_Response(Problem,Population,LastDec,LastObj,sampleN,mu,dim,gamma)
    D = Problem.D;
    N = Problem.N;
    Lower = Problem.lower;
    Upper = Problem.upper;
    sampleN = max(sampleN,Problem.M+2);

    Xs = rand(sampleN,D).*repmat(Upper-Lower,sampleN,1) + repmat(Lower,sampleN,1);
    FsPop = Problem.Evaluation(Xs);
    Fs = FsPop.objs';

    Xt = rand(sampleN,D).*repmat(Upper-Lower,sampleN,1) + repmat(Lower,sampleN,1);
    FaPop = Problem.Evaluation(Xt);
    Fa = FaPop.objs';

    POF = LastObj';
    try
        W = TrDMOEA_getW(Fs,Fa,mu,dim,'Gaussian',gamma);
        POFDeduced = TrDMOEA_getNewY(Fs,Fa,POF,W,'Gaussian',gamma);
        InitDec = zeros(N,D);
        baseDec = LastDec;
        if size(baseDec,1) < N
            baseDec = [baseDec;Population.decs];
        end
        for i = 1 : N
            target = POFDeduced(:,min(i,size(POFDeduced,2)));
            start  = baseDec(min(i,size(baseDec,1)),:);
            InitDec(i,:) = TrDMOEA_LocalSearch(Problem,Fs,Fa,W,target,start,Lower,Upper,gamma);
        end
    catch
        shift   = mean(Population.decs,1) - mean(LastDec,1);
        InitDec = Population.decs + repmat(shift,N,1) + randn(N,D).*repmat((Upper-Lower)*0.01,N,1);
    end
    InitDec = min(max(InitDec,repmat(Lower,N,1)),repmat(Upper,N,1));
end

function Dec = TrDMOEA_LocalSearch(Problem,Fs,Fa,W,target,start,Lower,Upper,gamma)
    D = Problem.D;
    if exist('fmincon','file') == 2
        options = optimset('display','off','MaxFunEvals',200,'MaxIter',50);
        try
            Dec = fmincon(@(x)TrDMOEA_Distance(Problem,Fs,Fa,W,x,target,gamma),start,[],[],[],[],Lower,Upper,[],options);
            return;
        catch
        end
    end
    Dec = start;
    best = TrDMOEA_Distance(Problem,Fs,Fa,W,Dec,target,gamma);
    sigma = 0.1*(Upper-Lower);
    for i = 1 : 20
        trial = Dec + randn(1,D).*sigma;
        trial = min(max(trial,Lower),Upper);
        value = TrDMOEA_Distance(Problem,Fs,Fa,W,trial,target,gamma);
        if value < best
            Dec  = trial;
            best = value;
        end
        sigma = sigma*0.9;
    end
end

function distance = TrDMOEA_Distance(Problem,Fs,Fa,W,Dec,target,gamma)
    Pop = Problem.Evaluation(Dec);
    Obj = Pop.objs';
    Y   = TrDMOEA_getNewY(Fs,Fa,Obj,W,'Gaussian',gamma);
    distance = sum((Y-target).^2);
end

function W = TrDMOEA_getW(Xs,Xt,mu,dim,kind,p1)
    n1 = size(Xs,2);
    n2 = size(Xt,2);
    X  = [Xs,Xt];
    K  = zeros(n1+n2);
    for i = 1 : n1+n2
        for j = 1 : n1+n2
            K(i,j) = TrDMOEA_getKernel(X(:,i),X(:,j),kind,p1);
        end
    end
    L = zeros(n1+n2);
    L(1:n1,1:n1) = 1/(n1*n1);
    L(n1+1:end,n1+1:end) = 1/(n2*n2);
    L(1:n1,n1+1:end) = -1/(n1*n2);
    L(n1+1:end,1:n1) = -1/(n1*n2);
    H = eye(n1+n2) - ones(n1+n2)/(n1+n2);
    Temp = (eye(n1+n2)+mu*K*L*K)\(K*H*K);
    [V,D] = eig(Temp);
    V = real(V);
    D = real(diag(D));
    [~,rank] = sort(D,'descend');
    count = min(max(1,dim),length(rank));
    W = V(:,rank(1:count));
end

function Y = TrDMOEA_getNewY(Xs,Xt,X,W,kind,p1)
    n1 = size(Xs,2);
    n2 = size(Xt,2);
    n3 = size(X,2);
    K  = zeros(n1+n2,n3);
    for j = 1 : n3
        for i = 1 : n1
            K(i,j) = TrDMOEA_getKernel(Xs(:,i),X(:,j),kind,p1);
        end
        for i = 1 : n2
            K(i+n1,j) = TrDMOEA_getKernel(Xt(:,i),X(:,j),kind,p1);
        end
    end
    Y = W'*K;
end

function k = TrDMOEA_getKernel(a,b,kind,p1)
    if strcmp(kind,'Gaussian')
        c = a-b;
        k = exp(-p1*(c'*c));
    elseif strcmp(kind,'Laplacian')
        c = a-b;
        k = exp(-p1*sqrt(c'*c));
    else
        k = a'*b;
    end
end
