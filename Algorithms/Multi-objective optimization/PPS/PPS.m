classdef PPS < ALGORITHM
% <2022> <multi> <real> <constrained/none> <dynamic>
% Population prediction strategy
% K  ---   5 --- Number of local PCA clusters
% p  ---   2 --- Order of the autoregressive predictor
% W  ---  23 --- Length of the historical window
% zeta --- 0.5 --- Ratio of random solutions for the first responses

% This implementation is adapted to the PlatEMO algorithm interface from
% the PPS code in VARE-main/PPS.

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [K,p,W,zeta] = Algorithm.ParameterSet(5,2,23,0.5);
            Algorithm.save = sign(Algorithm.save)*inf;

            %% Generate random population
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
            AllPop  = [];
            Archive = [];
            Centers = zeros(0,Problem.D);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                if PPS_Changed(Problem,Population)
                    AllPop  = [AllPop,Population];
                    Archive = [Archive,Population];
                    Centers = [Centers;mean(Population.decs,1)];
                    Population = PPS_Response(Problem,Archive,Centers,Problem.N,p,W,zeta);
                    [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
                else
                    OffDec = PPS_RMMEDAOperator(Population.decs,Problem.M,K);
                    OffDec = min(max(OffDec,repmat(Problem.lower,Problem.N,1)),repmat(Problem.upper,Problem.N,1));
                    Offspring = Problem.Evaluation(OffDec);
                    [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
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

function changed = PPS_Changed(Problem,Population)
    RePop1  = Population(randperm(end,ceil(end/10)));
    RePop2  = Problem.Evaluation(RePop1.decs);
    changed = ~isequal(RePop1.objs,RePop2.objs) || ~isequal(RePop1.cons,RePop2.cons);
end

function Population = PPS_Response(Problem,Archive,Centers,N,p,W,zeta)
    D   = Problem.D;
    Rch = size(Centers,1);
    if Rch <= p || length(Archive) < 2*N
        first = max(1,length(Archive)-N+1);
        Dec   = Archive(first:end).decs;
        nr  = max(1,round(zeta*N));
        idx = randperm(N,nr);
        Dec(idx,:) = rand(nr,D).*repmat(Problem.upper-Problem.lower,nr,1) + repmat(Problem.lower,nr,1);
    else
        Len       = min(W,Rch);
        History   = Centers(Rch-Len+1:Rch,:);
        NewCenter = zeros(1,D);
        for d = 1 : D
            series = History(:,d);
            if exist('ar','file') == 2 && exist('iddata','file') == 2
                try
                    model = ar(series,p,'fb');
                    data  = iddata(series);
                    last  = numel(series);
                    pred  = predict(data(max(1,last-p):last),model,1);
                    mse   = model.Report.Fit.MSE;
                    NewCenter(d) = pred.OutputData(end) + randn*sqrt(max(mse,eps));
                catch
                    NewCenter(d) = PPS_LinearPredict(series);
                end
            else
                NewCenter(d) = PPS_LinearPredict(series);
            end
        end
        PreDec = Archive(end-2*N+1:end-N).decs;
        CurDec = Archive(end-N+1:end).decs;
        CurManifold = CurDec - mean(CurDec,1);
        PreManifold = PreDec - mean(PreDec,1);
        minDistance = PPS_MinDistance(CurManifold,PreManifold);
        sigma       = mean(minDistance)/max(D,1);
        Dec      = CurManifold + repmat(NewCenter,N,1) + randn(N,D)*sigma;
        Dec      = min(max(Dec,repmat(Problem.lower,N,1)),repmat(Problem.upper,N,1));
    end
    Population = Problem.Evaluation(Dec);
end

function minDistance = PPS_MinDistance(PopDecA,PopDecB)
    minDistance = inf(size(PopDecA,1),1);
    for i = 1 : size(PopDecA,1)
        distance = sqrt(sum((PopDecB-repmat(PopDecA(i,:),size(PopDecB,1),1)).^2,2));
        minDistance(i) = min(distance);
    end
end

function value = PPS_LinearPredict(series)
    if numel(series) < 2
        value = series(end);
    else
        value = series(end) + (series(end)-series(end-1));
    end
end

function OffDec = PPS_RMMEDAOperator(PopDec,M,K)
    [N,D] = size(PopDec);
    K     = min(K,N);
    [Model,probability] = PPS_LocalPCA(PopDec,M,K);
    OffDec = zeros(N,D);
    for i = 1 : N
        k = find(rand<=probability,1);
        if isempty(k); k = K; end
        if ~isempty(Model(k).eVector)
            lower = Model(k).a - 0.25*(Model(k).b-Model(k).a);
            upper = Model(k).b + 0.25*(Model(k).b-Model(k).a);
            trial = rand(1,M-1).*(upper-lower) + lower;
            sigma = sum(abs(Model(k).eValue(M:D)))/max(D-M+1,1);
            OffDec(i,:) = Model(k).mean + trial*Model(k).eVector(:,1:M-1)' + randn(1,D)*sqrt(max(sigma,eps));
        else
            OffDec(i,:) = Model(k).mean + randn(1,D)*1e-2;
        end
    end
end

function [Model,probability] = PPS_LocalPCA(PopDec,M,K)
    [N,D] = size(PopDec);
    Model = struct('mean',num2cell(PopDec(randperm(N,K),:),2),'PI',eye(D),'eVector',[],'eValue',[],'a',[],'b',[]);
    partition = ones(N,1);
    for iter = 1 : 50
        distance = zeros(N,K);
        for k = 1 : K
            distance(:,k) = sum((PopDec-repmat(Model(k).mean,N,1))*Model(k).PI.*(PopDec-repmat(Model(k).mean,N,1)),2);
        end
        [~,partition] = min(distance,[],2);
        updated = false(1,K);
        for k = 1 : K
            oldMean = Model(k).mean;
            current = partition == k;
            if sum(current) < 2
                if ~any(current); current = randi(N); end
                Model(k).mean    = PopDec(current,:);
                Model(k).PI      = eye(D);
                Model(k).eVector = [];
                Model(k).eValue  = [];
            else
                Model(k).mean    = mean(PopDec(current,:),1);
                [eVector,eValue] = eig(cov(PopDec(current,:)-repmat(Model(k).mean,sum(current),1)));
                [eValue,rank]    = sort(diag(eValue),'descend');
                Model(k).eValue  = real(eValue);
                Model(k).eVector = real(eVector(:,rank));
                Model(k).PI      = Model(k).eVector(:,M:end)*Model(k).eVector(:,M:end)';
            end
            updated(k) = ~any(current) || norm(oldMean-Model(k).mean) > 1e-5;
        end
        if ~any(updated); break; end
    end
    for k = 1 : K
        if ~isempty(Model(k).eVector)
            hyperRectangle = (PopDec(partition==k,:)-repmat(Model(k).mean,sum(partition==k),1))*Model(k).eVector(:,1:M-1);
            Model(k).a = min(hyperRectangle,[],1);
            Model(k).b = max(hyperRectangle,[],1);
        else
            Model(k).a = zeros(1,M-1);
            Model(k).b = zeros(1,M-1);
        end
    end
    volume = prod(max(cat(1,Model.b)-cat(1,Model.a),eps),2);
    if sum(volume) == 0 || any(isnan(volume))
        probability = cumsum(ones(K,1)/K);
    else
        probability = cumsum(volume/sum(volume));
    end
end
