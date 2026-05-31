classdef FAMM_A3 < ALGORITHM
% <multi> <real> <dynamic>
% Memristive-matrix predictive FAMM for dynamic MOPs
% 将原先单一忆阻电荷调参改为“忆阻矩阵”机制：矩阵按维度存储历史位移、
% 方向一致性和挥发性强度，并用于中心/轨迹预测与搜索算子自适应。
%
% alpha0     --- 0.48 --- Base randomization strength
% beta0      --- 1.65 --- Base attractiveness
% gamma0     --- 0.80 --- Light absorption coefficient
% lambdaMem  --- 0.92 --- Memristive matrix decay factor
% etaMem     --- 0.35 --- Learning rate of each matrix write
% histLen    --- 6    --- Number of historical directions kept in the matrix
% eliteRate  --- 0.25 --- Ratio of elites used as attractors

%------------------------------- Note ------------------------------------
% The dynamic response borrows common prediction ideas in dynamic MOEAs:
% center-based prediction, second-order trend/acceleration prediction,
% memory archive reuse, and random immigrants.  The predicted direction is
% no longer controlled by a scalar charge; instead, a H-by-D memristive
% matrix stores recent useful shifts in decision space.  Its conductance
% vector gives dimension-wise confidence, allowing the operator to move more
% strongly along historically consistent dimensions while keeping diversity
% in uncertain dimensions.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [alpha0,beta0,gamma0,lambdaMem,etaMem,histLen,eliteRate] = ...
                Algorithm.ParameterSet(0.48,1.65,0.80,0.92,0.35,6,0.25);
            histLen = max(3,round(histLen));

            Algorithm.save = sign(Algorithm.save)*inf;

            %% Initialization
            Population = Problem.Initialization();
            [Population,Archive,Fitness] = FAMM_EnvironmentalSelection(Population,Problem.N);
            mem       = FAMM_A2_InitMemMatrix(Problem,Population.decs,histLen);
            AllPop    = [];
            prevFit   = mean(Fitness);
            stableGen = 0;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                %% Detect environmental change and perform matrix prediction response
                if FAMM_Changed(Problem,Population)
                    AllPop = [AllPop,Population];
                    [Population,Archive,Fitness,mem] = FAMM_A2_ChangeResponse(...
                        Problem,Population,Archive,mem,lambdaMem,etaMem);
                    stableGen = 0;
                    prevFit   = mean(Fitness);
                else
                    stableGen = stableGen + 1;
                end

                %% Update memristive matrix by successful search drift in stable periods
                currFit     = mean(Fitness);
                improvement = (prevFit-currFit)./max(abs(prevFit),1e-6);
                mem         = FAMM_A2_UpdateStableMemory(...
                    Problem,Population,Archive,mem,lambdaMem,etaMem,improvement);
                prevFit     = currFit;

                %% Matrix-state driven parameter scheduling
                g            = mem.global;
                stableSignal = 1 - exp(-stableGen/20);
                alpha        = alpha0*(0.18 + 0.72*(1-g) + 0.18*(1-stableSignal));
                beta         = beta0 *(0.78 + 0.42*g);
                gamma        = gamma0*(0.45 + 0.65*(1-g));
                pPred        = 0.22 + 0.48*g + 0.12*stableSignal;
                pPred        = max(0.18,min(0.82,pPred));
                eliteRateNow = eliteRate*(0.75 + 0.85*stableSignal + 0.45*g);
                eliteRateNow = max(0.12,min(0.45,eliteRateNow));

                %% Memristive predictive firefly-differential search
                Offspring = FAMM_A2_OperatorMemristive(Problem,Population,Archive,Fitness,...
                    mem,alpha,beta,gamma,eliteRateNow,pPred);

                [Population,Archive,Fitness] = FAMM_EnvironmentalSelection([Population,Offspring],Problem.N);

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

function mem = FAMM_A2_InitMemMatrix(Problem,Dec,histLen)
% Initialize an H-by-D memristive matrix.  Row = one historical direction;
% column = one decision variable.  Conductance is dimension-wise confidence.

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

function [Population,Archive,Fitness,mem] = FAMM_A2_ChangeResponse(Problem,Population,Archive,mem,lambdaMem,etaMem)
% Environmental response using the memristive matrix.  Re-evaluated elites
% give the observed shift; the matrix predicts the next useful direction.

    N     = Problem.N;
    D     = Problem.D;
    lower = Problem.lower;
    upper = Problem.upper;
    span  = max(upper-lower,1e-12);

    % Re-evaluate current population and historical archive in the new environment.
    Population = Problem.Evaluation(Population.decs);
    if ~isempty(Archive)
        Archive = Problem.Evaluation(Archive.decs);
        BaseAll = [Population,Archive];
    else
        BaseAll = Population;
    end
    [BasePop,BaseArchive,~] = FAMM_EnvironmentalSelection(BaseAll,N);

    % Center-based observation: promising region before/after the change.
    newCenter = FAMM_A2_EliteCenter(BasePop,BaseArchive);
    observed  = newCenter - mem.center;
    quality   = min(1,0.35 + 3*norm(observed./span)/sqrt(D));
    mem       = FAMM_A2_WriteMemory(mem,observed,span,quality,lambdaMem,etaMem);
    predShift = 0.55*observed + 0.45*FAMM_A2_PredictShift(mem,span);
    if norm(predShift./span) < 1e-8
        predShift = observed;
    end
    predShift    = max(min(predShift,0.35*span),-0.35*span);
    mem.velocity = predShift;

    % Candidate composition: predicted archive/elite solutions, directional
    % differential samples, perturbed survivors, opposition samples and random immigrants.
    g = mem.global;
    G = mem.conductance;
    if ~isempty(BaseArchive)
        GuideDec = BaseArchive.decs;
    else
        GuideDec = BasePop.decs;
    end
    PopDec = BasePop.decs;

    nImm  = round(0.10*g*N);
    nOpp  = round(0.10*g*N);
    nDiff = round(0.15*N);
    nPred = max(1,round((0.38+0.24*g)*N));
    nMut  = N - nImm - nOpp - nDiff - nPred;
    if nMut < 0
        nPred = max(1,nPred+nMut);
        nMut  = 0;
    end

    Cand = [];

    % 1) Center/trajectory prediction, similar to centroid prediction DMOEAs.
    if nPred > 0
        id   = randi(size(GuideDec,1),nPred,1);
        coef = 0.65 + 0.70*rand(nPred,1);
        Noise = randn(nPred,D).*repmat(span,nPred,1).*repmat(0.01+0.05*(1-G),nPred,1);
        Pred  = GuideDec(id,:) + repmat(coef,1,D).*repmat(predShift,nPred,1).*repmat(0.35+0.65*G,nPred,1) + Noise;
        Cand  = [Cand;Pred];
    end

    % 2) Memristive directional differential samples: exploit reliable dimensions.
    if nDiff > 0
        ADec = [GuideDec;PopDec];
        r1   = randi(size(ADec,1),nDiff,1);
        r2   = randi(size(ADec,1),nDiff,1);
        r3   = randi(size(ADec,1),nDiff,1);
        Mask = rand(nDiff,D) < repmat(0.20+0.75*G,nDiff,1);
        F    = 0.25 + 0.45*rand(nDiff,1);
        Diff = ADec(r1,:) + repmat(F,1,D).*(ADec(r2,:)-ADec(r3,:)).*Mask + 0.45*repmat(predShift,nDiff,1);
        Cand = [Cand;Diff];
    end

    % 3) Perturbed survivors preserve convergence information.
    if nMut > 0
        id  = randi(size(PopDec,1),nMut,1);
        Mut = PopDec(id,:) + randn(nMut,D).*repmat(span,nMut,1).*repmat(0.02+0.08*(1-G),nMut,1);
        Cand = [Cand;Mut];
    end

    % 4) Opposition-based and random immigrants improve coverage after severe shifts.
    if nOpp > 0
        id  = randi(size(PopDec,1),nOpp,1);
        Opp = repmat(lower+upper,nOpp,1) - PopDec(id,:) + 0.03*randn(nOpp,D).*repmat(span,nOpp,1);
        Cand = [Cand;Opp];
    end
    if nImm > 0
        Imm  = repmat(lower,nImm,1) + rand(nImm,D).*repmat(span,nImm,1);
        Cand = [Cand;Imm];
    end

    Cand      = FAMM_A2_Repair(Problem,Cand);
    Offspring = Problem.Evaluation(Cand);
    [Population,Archive,Fitness] = FAMM_EnvironmentalSelection([BasePop,Offspring],N);
    mem.center = FAMM_A2_EliteCenter(Population,Archive);
end

function mem = FAMM_A2_UpdateStableMemory(Problem,Population,Archive,mem,lambdaMem,etaMem,improvement)
% During unchanged periods, successful population drift is written into the
% matrix with a smaller quality.  This lets the matrix remember not only
% environment jumps but also useful local search directions.

    span      = max(Problem.upper-Problem.lower,1e-12);
    newCenter = FAMM_A2_EliteCenter(Population,Archive);
    shift     = newCenter - mem.center;
    normShift = norm(shift./span)/sqrt(Problem.D);

    if normShift > 1e-7 && improvement > -0.05
        quality = 0.08 + 0.55*min(1,max(0,improvement)*10) + 0.25*min(1,normShift*8);
        mem     = FAMM_A2_WriteMemory(mem,shift,span,quality,lambdaMem,etaMem);
    else
        mem.Q = lambdaMem*mem.Q;
        mem   = FAMM_A2_UpdateConductance(mem);
    end
    mem.center   = newCenter;
    mem.velocity = FAMM_A2_PredictShift(mem,span);
end

function mem = FAMM_A2_WriteMemory(mem,shift,span,quality,lambdaMem,etaMem)
% One write operation to the memristive matrix.  Old rows fade through Q,
% while the new row is blended by etaMem, mimicking nonvolatile conductance.

    if any(~isfinite(shift)) || any(~isfinite(span))
        mem.Q = lambdaMem*mem.Q;
        mem   = FAMM_A2_UpdateConductance(mem);
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
    mem          = FAMM_A2_UpdateConductance(mem);
end

function mem = FAMM_A2_UpdateConductance(mem)
% Convert matrix history into a conductance vector.  High conductance means
% a dimension has large and sign-consistent historical movement.

    active = find(mem.Q > 1e-12);
    if isempty(active)
        mem.predNorm    = zeros(1,size(mem.M,2));
        mem.conductance = 0.05*ones(1,size(mem.M,2));
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

function shift = FAMM_A2_PredictShift(mem,span)
% First/second-order matrix prediction.  The weighted matrix mean is the
% base velocity and the difference between the two most recent rows provides
% an acceleration term, a common idea in prediction-based DMOEAs.

    predNorm = mem.predNorm;
    recent   = FAMM_A2_RecentRows(mem,2);
    if size(recent,1) >= 2
        accel    = recent(1,:) - recent(2,:);
        predNorm = 0.72*predNorm + 0.28*recent(1,:) + 0.35*mem.global*accel;
    elseif size(recent,1) == 1
        predNorm = 0.65*predNorm + 0.35*recent(1,:);
    end
    predNorm = max(min(predNorm,0.35),-0.35);
    shift    = predNorm.*span;
end

function rows = FAMM_A2_RecentRows(mem,k)
% Return up to k most recent valid matrix rows, newest first.

    n    = min([k,mem.count,mem.histLen]);
    rows = zeros(0,size(mem.M,2));
    for t = 1:n
        idx = mod(mem.ptr-1-t,mem.histLen) + 1;
        if mem.Q(idx) > 1e-12
            rows = [rows;mem.M(idx,:)]; %#ok<AGROW>
        end
    end
end

function center = FAMM_A2_EliteCenter(Population,Archive)
% Robust elite center used by both prediction and stable search learning.

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

function Offspring = FAMM_A2_OperatorMemristive(Problem,Population,Archive,Fitness,mem,alpha,beta0,gamma,eliteRate,pPred)
% Memristive predictive firefly-differential operator.
% - Firefly attraction keeps convergence pressure.
% - Matrix predicted velocity tracks future POS movement.
% - Dimension-wise conductance gates differential variation and noise.

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
    EliteC   = mean(EliteDec,1);

    if ~isempty(Archive)
        ADec = [Archive.decs;EliteDec];
    else
        ADec = EliteDec;
    end
    if isempty(ADec)
        ADec = Dec;
    end

    predShift = max(min(mem.velocity,0.30*span),-0.30*span);
    G         = max(0.02,min(0.98,mem.conductance));
    g         = mem.global;

    OffDec = zeros(N,D);
    for i = 1:N
        xi = Dec(i,:);

        %% Firefly attraction to brighter/elite individuals
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

        %% Predictive elite guide and dimension-gated differential vector
        xElite = EliteDec(randi(size(EliteDec,1)),:);
        xGuide = xElite + (0.55+0.75*rand)*predShift;
        rij    = norm((xi-xGuide)./span)/sqrt(D);
        betaG  = min(1.45,beta0*exp(-0.5*gamma*rij^2));

        r1   = randi(size(ADec,1));
        r2   = randi(size(ADec,1));
        diff = ADec(r1,:) - ADec(r2,:);
        mask = rand(1,D) < (0.18 + 0.78*G);
        F    = 0.20 + 0.35*(1-g) + 0.20*rand;

        if rand < pPred
            % Prediction mode: confident dimensions follow the matrix trend.
            memDir = predShift.*(0.30+0.70*G);
            noise  = alpha*randn(1,D).*span.*(0.012+0.045*(1-G));
            step   = 0.55*move + betaG*(xGuide-xi) + (0.35+0.45*g)*memDir + 0.20*F*(diff.*mask) + noise;
        else
            % Exploration/tracking mode: Lévy/Gaussian disturbance is stronger
            % on low-conductance dimensions, avoiding premature collapse.
            if rand < 0.5
                randStep = FAMM_A2_Levy(1,D).*span.*(0.018+0.055*(1-G));
            else
                randStep = randn(1,D).*span.*(0.020+0.065*(1-G));
            end
            centerPull = 0.06*(0.5+g)*(EliteC-xi);
            step = move + 0.35*F*(diff.*mask) + 0.25*g*predShift.*(0.20+0.80*G) + centerPull + alpha*randStep;
        end
        OffDec(i,:) = xi + step;
    end

    OffDec    = FAMM_A2_Repair(Problem,OffDec);
    Offspring = Problem.Evaluation(OffDec);
end

function OffDec = FAMM_A2_Repair(Problem,OffDec)
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

function L = FAMM_A2_Levy(n,d)
% Mantegna Lévy flight with beta = 1.5.

    beta  = 1.5;
    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u     = randn(n,d)*sigma;
    v     = randn(n,d);
    L     = u./(abs(v).^(1/beta)+eps);
    L     = max(min(L,10),-10);
end
