classdef FAMM_A4 < ALGORITHM
% <multi> <real> <dynamic>
% Simplified memristive-matrix firefly algorithm for dynamic MOPs
%
% alpha0     --- 0.50 --- Base randomization strength
% beta0      --- 1.60 --- Base attractiveness
% gamma0     --- 0.85 --- Light absorption coefficient
% lambdaMem  --- 0.92 --- Memristive matrix decay factor
% etaMem     --- 0.35 --- Learning rate of each matrix write
% histLen    --- 6    --- Number of historical directions kept
% eliteRate  --- 0.25 --- Ratio of elite firefly attractors

%------------------------------- Note ------------------------------------
% A4 is a compact ablation version of A3.  It keeps only two mechanisms:
% 1) a dimension-wise memristive matrix that stores environmental shifts;
% 2) a firefly search operator guided by elites and the matrix prediction.
% Differential sampling, opposition learning, random immigrants, Levy flight,
% and stable-period memory writing are intentionally removed.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [alpha0,beta0,gamma0,lambdaMem,etaMem,histLen,eliteRate] = ...
                Algorithm.ParameterSet(0.50,1.60,0.85,0.92,0.35,6,0.25);
            histLen = max(3,round(histLen));

            Algorithm.save = sign(Algorithm.save)*inf;

            %% Initialization
            Population = Problem.Initialization();
            [Population,Archive,Fitness] = FAMM_EnvironmentalSelection(Population,Problem.N);
            mem    = FAMM_A4_InitMemMatrix(Problem,Population.decs,histLen);
            AllPop = [];

            %% Optimization
            while Algorithm.NotTerminated(Population)
                %% Detect environmental change and update the memristive matrix
                if FAMM_Changed(Problem,Population)
                    AllPop = [AllPop,Population];
                    [Population,Archive,Fitness,mem] = FAMM_A4_ChangeResponse(...
                        Problem,Population,Archive,mem,lambdaMem,etaMem);
                end

                %% Matrix-state driven firefly parameters
                g     = mem.global;
                alpha = alpha0*(0.25 + 0.75*(1-g));
                beta  = beta0 *(0.85 + 0.30*g);
                gamma = gamma0*(0.55 + 0.65*(1-g));

                %% Memristive firefly search
                Offspring = FAMM_A4_OperatorFirefly(Problem,Population,Archive,Fitness,...
                    mem,alpha,beta,gamma,eliteRate);

                [Population,Archive,Fitness] = FAMM_EnvironmentalSelection([Population,Offspring],Problem.N);
                mem.center = FAMM_A4_EliteCenter(Population,Archive);

                %% Return all populations for dynamic metric calculation
                if Problem.FE >= Problem.maxFE
                    Population = [AllPop,Population];
                    [~,rank]   = sort(Population.adds(zeros(length(Population),1)));
                    Population = Population(rank);
                end
            end
        end
    end
end

function mem = FAMM_A4_InitMemMatrix(Problem,Dec,histLen)
% Initialize an H-by-D matrix.  Each row stores one historical shift and each
% column records the confidence of one decision variable.

    D = Problem.D;
    mem.M           = zeros(histLen,D);
    mem.Q           = zeros(histLen,1);
    mem.ptr         = 1;
    mem.count       = 0;
    mem.histLen     = histLen;
    mem.center      = mean(Dec,1);
    mem.predNorm    = zeros(1,D);
    mem.velocity    = zeros(1,D);
    mem.conductance = 0.05*ones(1,D);
    mem.global      = 0.05;
end

function [Population,Archive,Fitness,mem] = FAMM_A4_ChangeResponse(Problem,Population,Archive,mem,lambdaMem,etaMem)
% Re-evaluate elites in the new environment and write the observed center
% shift into the memristive matrix.  No immigrants, opposition, or DE samples
% are generated in this simplified version.

    N    = Problem.N;
    D    = Problem.D;
    span = max(Problem.upper-Problem.lower,1e-12);

    Population = Problem.Evaluation(Population.decs);
    if ~isempty(Archive)
        Archive = Problem.Evaluation(Archive.decs);
        BaseAll = [Population,Archive];
    else
        BaseAll = Population;
    end

    [Population,Archive,Fitness] = FAMM_EnvironmentalSelection(BaseAll,N);
    newCenter = FAMM_A4_EliteCenter(Population,Archive);
    observed  = newCenter - mem.center;
    quality   = min(1,0.20 + 4*norm(observed./span)/sqrt(D));

    mem          = FAMM_A4_WriteMemory(mem,observed,span,quality,lambdaMem,etaMem);
    mem.velocity = FAMM_A4_PredictShift(mem,span);
    mem.center   = newCenter;
end

function mem = FAMM_A4_WriteMemory(mem,shift,span,quality,lambdaMem,etaMem)
% Write one normalized environmental shift into the matrix.

    if any(~isfinite(shift)) || any(~isfinite(span))
        mem.Q = lambdaMem*mem.Q;
        mem   = FAMM_A4_UpdateConductance(mem);
        return;
    end

    v = shift./span;
    v = max(min(v,0.50),-0.50);

    mem.Q = lambdaMem*mem.Q;
    row   = mem.ptr;
    mem.M(row,:) = (1-etaMem)*mem.M(row,:) + etaMem*v;
    mem.Q(row)   = max(1e-6,min(1,quality));
    mem.ptr      = mod(mem.ptr,mem.histLen) + 1;
    mem.count    = min(mem.histLen,mem.count+1);
    mem          = FAMM_A4_UpdateConductance(mem);
end

function mem = FAMM_A4_UpdateConductance(mem)
% High conductance means a variable has large and sign-consistent historical
% movement, so the firefly operator can trust the matrix prediction there.

    active = find(mem.Q > 1e-12);
    D      = size(mem.M,2);
    if isempty(active)
        mem.predNorm    = zeros(1,D);
        mem.conductance = 0.05*ones(1,D);
        mem.global      = 0.05;
        return;
    end

    W = mem.Q(active)./sum(mem.Q(active));
    H = mem.M(active,:);

    meanV       = W'*H;
    meanAbs     = W'*abs(H);
    consistency = abs(W'*sign(H));
    G           = (1-exp(-12*meanAbs)).*(0.25+0.75*consistency);
    G           = max(0.02,min(0.98,G));

    mem.predNorm    = meanV;
    mem.conductance = G;
    mem.global      = max(0.02,min(0.98,mean(G)));
end

function shift = FAMM_A4_PredictShift(mem,span)
% Matrix prediction with a small acceleration term from the two latest rows.

    predNorm = mem.predNorm;
    recent   = FAMM_A4_RecentRows(mem,2);
    if size(recent,1) >= 2
        accel    = recent(1,:) - recent(2,:);
        predNorm = 0.75*predNorm + 0.25*recent(1,:) + 0.30*mem.global*accel;
    elseif size(recent,1) == 1
        predNorm = 0.70*predNorm + 0.30*recent(1,:);
    end

    predNorm = max(min(predNorm,0.35),-0.35);
    shift    = predNorm.*span;
end

function rows = FAMM_A4_RecentRows(mem,k)
% Return up to k valid matrix rows, newest first.

    n    = min([k,mem.count,mem.histLen]);
    rows = zeros(0,size(mem.M,2));
    for t = 1:n
        idx = mod(mem.ptr-1-t,mem.histLen) + 1;
        if mem.Q(idx) > 1e-12
            rows = [rows;mem.M(idx,:)]; %#ok<AGROW>
        end
    end
end

function center = FAMM_A4_EliteCenter(Population,Archive)
% Use archived nondominated solutions when available; otherwise use the first
% nondominated front of the current population.

    if nargin > 1 && ~isempty(Archive)
        Dec = Archive.decs;
        if size(Dec,1) > 0
            center = mean(Dec,1);
            return;
        end
    end

    N = length(Population);
    try
        [FrontNo,~] = NDSort(Population.objs,Population.cons,N);
    catch
        [FrontNo,~] = NDSort(Population.objs,N);
    end

    first = find(FrontNo == min(FrontNo));
    if isempty(first)
        center = mean(Population.decs,1);
    else
        center = mean(Population(first).decs,1);
    end
end

function Offspring = FAMM_A4_OperatorFirefly(Problem,Population,Archive,Fitness,mem,alpha,beta0,gamma,eliteRate)
% Firefly search guided by Pareto rank, elite solutions, and the memristive
% matrix prediction.  This is the only variation operator in A4.

    N     = length(Population);
    D     = Problem.D;
    lower = Problem.lower;
    upper = Problem.upper;
    span  = max(upper-lower,1e-12);
    Dec   = Population.decs;

    try
        [FrontNo,~] = NDSort(Population.objs,Population.cons,N);
    catch
        [FrontNo,~] = NDSort(Population.objs,N);
    end
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    CrowdDis(isnan(CrowdDis)) = 0;
    finiteCD = CrowdDis(isfinite(CrowdDis));
    if isempty(finiteCD)
        finiteCD = 1;
    end
    CrowdDis(isinf(CrowdDis)) = max(finiteCD) + 1;

    if isempty(Fitness) || length(Fitness) ~= N
        Fitness = FAMM_CalFitness(Population.objs);
    end

    nElite   = max(2,min(N,ceil(eliteRate*N)));
    [~,rank] = sortrows([FrontNo(:),Fitness(:),-CrowdDis(:)]);
    EliteIdx = rank(1:nElite);
    EliteDec = Dec(EliteIdx,:);

    if ~isempty(Archive)
        GuideDec = Archive.decs;
    else
        GuideDec = EliteDec;
    end
    if isempty(GuideDec)
        GuideDec = EliteDec;
    end

    predShift = max(min(mem.velocity,0.30*span),-0.30*span);
    G         = max(0.02,min(0.98,mem.conductance));
    g         = mem.global;

    OffDec = zeros(N,D);
    for i = 1:N
        xi = Dec(i,:);

        better = find(FrontNo < FrontNo(i));
        if isempty(better)
            better = EliteIdx(:)';
        end
        better(better == i) = [];

        move = zeros(1,D);
        wsum = 0;
        if ~isempty(better)
            k   = min(3,length(better));
            sel = better(randperm(length(better),k));
            for s = 1:k
                xj  = Dec(sel(s),:);
                rij = norm((xi-xj)./span)/sqrt(D);
                bij = beta0*exp(-gamma*rij^2);
                move = move + bij*(xj-xi);
                wsum = wsum + bij;
            end
            if wsum > 0
                move = move/wsum;
            end
        end

        xElite = GuideDec(randi(size(GuideDec,1)),:);
        xGuide = xElite + predShift.*(0.30 + 0.70*G);
        rij    = norm((xi-xGuide)./span)/sqrt(D);
        betaG  = min(1.40,beta0*exp(-0.5*gamma*rij^2));

        memDir = predShift.*(0.20 + 0.80*G);
        noise  = alpha*randn(1,D).*span.*(0.015 + 0.055*(1-G));
        step   = move + betaG*(xGuide-xi) + 0.35*g*memDir + noise;

        OffDec(i,:) = xi + step;
    end

    OffDec    = FAMM_A4_Repair(Problem,OffDec);
    Offspring = Problem.Evaluation(OffDec);
end

function OffDec = FAMM_A4_Repair(Problem,OffDec)
% Bound handling and encoding repair.

    if isempty(OffDec)
        return;
    end

    Lower  = repmat(Problem.lower,size(OffDec,1),1);
    Upper  = repmat(Problem.upper,size(OffDec,1),1);
    OffDec = min(max(OffDec,Lower),Upper);
    index  = ismember(Problem.encoding,2:4);
    OffDec(:,index) = round(OffDec(:,index));
end
