classdef FAMM < ALGORITHM
% <2026> <multi> <real> <dynamic>
% Firefly Algorithm with Adaptive Memristive Memory for dynamic MOPs
% alpha0   --- 0.55 --- Base random walk strength of fireflies
% beta0    --- 1.80 --- Base attractiveness at zero distance
% gamma0   --- 1.00 --- Base light absorption coefficient
% rho      --- 0.35 --- Maximum ratio of memristive random immigrants
% archiveR --- 1.50 --- Size ratio of the external memory archive

%------------------------------- Reference --------------------------------
% The algorithm combines common ideas from recent dynamic MOEAs, including
% memory, center/trajectory prediction, random immigrants, and NSGA-III-like
% reference-vector diversity preservation, with a memristive state equation
% used as an adaptive nonvolatile controller for dynamic environments.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference
% "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB
% platform for evolutionary multi-objective optimization [educational
% forum], IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [alpha0,beta0,gamma0,rho,archiveR] = Algorithm.ParameterSet(0.55,1.80,1.00,0.35,1.50);
            Algorithm.save = sign(Algorithm.save)*inf;

            N        = Problem.N;
            D        = Problem.D;
            ArchiveN = max(N,ceil(archiveR*N));
            nCheck   = max(1,ceil(0.10*N));
            %% Initialization
            Population = Problem.Initialization();
            Archive    = FAMM_EnvironmentalSelection(Population,ArchiveN);
            AllPop     = [];

            % Memristor state. x is the internal state and g is normalized
            % conductance.  Large g means a recent/severe change and triggers
            % stronger exploration and response; small g means a stable
            % environment and strengthens convergence.
            mem.x       = 0.05;
            mem.g       = FAMM_Conductance(mem.x);
            mem.decay   = 0.90;
            mem.gain    = 0.80;
            mem.prevC   = mean(Population.decs,1);
            mem.currC   = mem.prevC;
            mem.velocity= zeros(1,D);
            stableGen   = 0;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                %% Detect and react to environmental changes
                [changed,severity,Population] = FAMM_DetectChange(Problem,Population,nCheck);
                if changed
                    AllPop = [AllPop,Population];

                    oldC       = mem.currC;
                    mem.currC  = mean(Population.decs,1);
                    observedV  = mem.currC - oldC;
                    mem        = FAMM_UpdateMemristor(mem,severity,observedV);
                    stableGen  = 0;

                    % Re-evaluate memory in the new environment, then create a
                    % memristor-controlled response population.
                    if ~isempty(Archive)
                        Archive = Problem.Evaluation(Archive.decs);
                    end
                    Population = FAMM_Response(Problem,Population,Archive,mem,N,rho);
                    Archive    = FAMM_EnvironmentalSelection([Archive,Population],ArchiveN);
                else
                    % Volatility naturally fades but is not forgotten instantly.
                    stableGen = stableGen + 1;
                    mem.x     = mem.decay*mem.x;
                    mem.g     = FAMM_Conductance(mem.x);
                end

                %% Main search: memristive multiobjective firefly operator
                progress = min(1,Problem.FE/Problem.maxFE);
                alpha    = alpha0*(0.10 + 0.90*(1-progress))*(0.35 + 0.85*mem.g);
                beta     = beta0*(0.65 + 0.35*(1-mem.g));
                gamma    = gamma0*(0.50 + 1.50*(1-mem.g));
                Offspring = FAMM_OperatorFirefly(Problem,Population,Archive,alpha,beta,gamma,stableGen);

                %% Elitist selection and memory update
                Population = FAMM_EnvironmentalSelection([Population,Offspring],N);
                Archive    = FAMM_EnvironmentalSelection([Archive,Population],ArchiveN);

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

function [changed,severity,Population] = FAMM_DetectChange(Problem,Population,nCheck)
% Detect changes by re-evaluating representative sentinels.  Elitist and
% widely spread individuals are preferred, making detection more reliable
% than purely random checking while keeping the FE cost low.

    N = length(Population);
    nCheck = min(nCheck,N);
    try
        [FrontNo,~] = NDSort(Population.objs,Population.cons,N);
    catch
        [FrontNo,~] = NDSort(Population.objs,N);
    end
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    CrowdDis(isnan(CrowdDis)) = 0;
    [~,rank] = sortrows([FrontNo(:),-CrowdDis(:)]);
    index    = rank(1:nCheck);

    oldObj  = Population(index).objs;
    oldCon  = Population(index).cons;
    checked = Problem.Evaluation(Population(index).decs);
    newObj  = checked.objs;
    newCon  = checked.cons;

    diffObj  = abs(newObj-oldObj)./max(1,abs(oldObj));
    diffCon  = abs(newCon-oldCon)./max(1,abs(oldCon));
    severity = mean([diffObj(:);diffCon(:)]);
    changed  = severity > 1e-6;
    if changed
        Population(index) = checked;
    end
end

function mem = FAMM_UpdateMemristor(mem,severity,observedV)
% Voltage-driven memristor update with a momentum-like trajectory memory.
% The state is bounded, nonlinear, and nonvolatile, mimicking conductance
% accumulation after consecutive changes.

    voltage = min(1,12*severity);
    mem.x   = min(1,max(0,mem.decay*mem.x + mem.gain*voltage*(1-mem.x)));
    mem.g   = FAMM_Conductance(mem.x);
    if all(isfinite(observedV))
        mem.velocity = (1-mem.g)*mem.velocity + mem.g*observedV;
    end
    mem.prevC = mem.currC;
end

function g = FAMM_Conductance(x)
% Normalized HP-type memristor conductance. Ron << Roff.

    Ron = 1;
    Roff = 12;
    R = Ron*x + Roff*(1-x);
    g = (1./R - 1/Roff)./(1/Ron - 1/Roff);
    g = min(1,max(0,g));
end

function Population = FAMM_Response(Problem,Population,Archive,mem,N,rho)
% Memristive response after environmental change.  The response mixes
% predicted memory, perturbed survivors, opposition solutions, and random
% immigrants.  The mixture is fully controlled by mem.g.

    D     = Problem.D;
    lower = Problem.lower;
    upper = Problem.upper;
    span  = upper - lower;
    Dec   = Population.decs;

    nImm  = ceil(rho*mem.g*N);
    nOpp  = ceil(0.15*mem.g*N);
    nPred = ceil(0.45*mem.g*N);
    nMut  = max(0,N-nImm-nOpp-nPred);
    Cand  = [];

    % 1) Memory/trajectory prediction from archived elite decisions.
    if nPred > 0
        if ~isempty(Archive)
            ADec = Archive.decs;
        else
            ADec = Dec;
        end
        id   = randi(size(ADec,1),nPred,1);
        Pred = ADec(id,:) + repmat(mem.velocity,nPred,1) + ...
               (0.01+0.04*mem.g)*randn(nPred,D).*repmat(span,nPred,1);
        Cand = [Cand;Pred];
    end

    % 2) Perturbed survivors preserve useful convergence information.
    if nMut > 0
        id  = randi(size(Dec,1),nMut,1);
        Mut = Dec(id,:) + (0.02+0.16*mem.g)*randn(nMut,D).*repmat(span,nMut,1);
        Cand = [Cand;Mut];
    end

    % 3) Opposition-based immigrants improve rapid coverage after shifts.
    if nOpp > 0
        id  = randi(size(Dec,1),nOpp,1);
        Opp = repmat(lower+upper,nOpp,1) - Dec(id,:) + ...
              0.05*randn(nOpp,D).*repmat(span,nOpp,1);
        Cand = [Cand;Opp];
    end

    % 4) Random immigrants are activated for severe/frequent changes.
    if nImm > 0
        Imm  = repmat(lower,nImm,1) + rand(nImm,D).*repmat(span,nImm,1);
        Cand = [Cand;Imm];
    end

    if isempty(Cand)
        Population = FAMM_EnvironmentalSelection(Population,N);
    else
        Cand = min(max(Cand,repmat(lower,size(Cand,1),1)),repmat(upper,size(Cand,1),1));
        Off  = Problem.Evaluation(Cand);
        Population = FAMM_EnvironmentalSelection([Population,Off],N);
    end
end

function Offspring = FAMM_OperatorFirefly(Problem,Population,Archive,alpha,beta0,gamma,stableGen)
% Multiobjective firefly movement. Brightness is determined by nondominated
% rank, crowding distance, and archive/reference guidance.  Each firefly is
% attracted by one or more brighter fireflies and an elite memory target,
% plus Gaussian/Lévy randomization.

    N     = length(Population);
    D     = Problem.D;
    lower = Problem.lower;
    upper = Problem.upper;
    span  = upper - lower;
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
    Bright = -FrontNo(:) + 1e-3*CrowdDis(:);

    if ~isempty(Archive)
        ADec = Archive.decs;
        AObj = Archive.objs;
    else
        ADec = Dec;
        AObj = Population.objs;
    end

    OffDec = zeros(N,D);
    for i = 1 : N
        xi = Dec(i,:);
        better = find(Bright > Bright(i));
        if isempty(better)
            better = find(FrontNo == min(FrontNo));
        end
        better(better==i) = [];

        move = zeros(1,D);
        wsum = 0;
        if ~isempty(better)
            k   = min(3,length(better));
            sel = better(randperm(length(better),k));
            for s = 1 : k
                xj  = Dec(sel(s),:);
                rij = norm((xi-xj)./max(span,eps))/sqrt(D);
                bij = beta0*exp(-gamma*rij^2);
                move = move + bij*(xj-xi);
                wsum = wsum + bij;
            end
        end
        if wsum > 0
            move = move/wsum;
        end

        % Memory elite target: choose an archived solution in a sparse
        % objective region; this improves convergence and distribution.
        elite = FAMM_SelectArchiveElite(AObj);
        xElite = ADec(elite,:);
        rij = norm((xi-xElite)./max(span,eps))/sqrt(D);
        betaE = 0.5*beta0*exp(-0.5*gamma*rij^2);

        if mod(i,2) == 0 || stableGen < 2
            randStep = FAMM_Levy(1,D).*span*0.03;
        else
            randStep = randn(1,D).*span*0.04;
        end
        OffDec(i,:) = xi + move + betaE*(xElite-xi) + alpha*randStep;
    end

    OffDec = FAMM_Repair(Problem,OffDec);
    Offspring = Problem.Evaluation(OffDec);
end

function index = FAMM_SelectArchiveElite(PopObj)
% Select an elite from a less crowded objective-space region.

    NA = size(PopObj,1);
    if NA == 1
        index = 1;
        return;
    end
    try
        [FrontNo,~] = NDSort(PopObj,inf);
    catch
        FrontNo = ones(1,NA);
    end
    CrowdDis = CrowdingDistance(PopObj,FrontNo);
    CrowdDis(isnan(CrowdDis)) = 0;
    finiteCD = CrowdDis(isfinite(CrowdDis));
    if isempty(finiteCD)
        finiteCD = 1;
    end
    CrowdDis(isinf(CrowdDis)) = max(finiteCD) + 1;
    first = find(FrontNo == min(FrontNo));
    score = CrowdDis(first) - 1e-6*rand(1,length(first));
    [~,best] = max(score);
    index = first(best);
end

function OffDec = FAMM_Repair(Problem,OffDec)
% Bound handling and encoding repair.

    Lower = repmat(Problem.lower,size(OffDec,1),1);
    Upper = repmat(Problem.upper,size(OffDec,1),1);
    OffDec = min(max(OffDec,Lower),Upper);
    index = ismember(Problem.encoding,2:4);
    OffDec(:,index) = round(OffDec(:,index));
end

function L = FAMM_Levy(n,d)
% Mantegna Lévy flight with beta = 1.5.

    beta = 1.5;
    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u = randn(n,d)*sigma;
    v = randn(n,d);
    L = u./(abs(v).^(1/beta)+eps);
    L = max(min(L,10),-10);
end

function Population = FAMM_EnvironmentalSelection(Population,N)
% NSGA-II/NSGA-III hybrid environmental selection. Nondominated sorting
% drives convergence, crowding/reference-vector niching keeps diversity.

    if isempty(Population)
        return;
    end
    if length(Population) <= N
        return;
    end

    PopObj = Population.objs;
    PopCon = Population.cons;
    try
        [FrontNo,MaxFNo] = NDSort(PopObj,PopCon,N);
    catch
        [FrontNo,MaxFNo] = NDSort(PopObj,N);
    end
    Next = FrontNo < MaxFNo;
    Last = find(FrontNo == MaxFNo);
    K    = N - sum(Next);
    if K > 0
        Choose = FAMM_LastSelection(PopObj,FrontNo,Last,K);
        Next(Last(Choose)) = true;
    end
    Population = Population(Next);
end

function Choose = FAMM_LastSelection(PopObj,FrontNo,Last,K)
% Select K solutions from the critical front. For two/three objectives it
% uses reference-vector niching; otherwise it falls back to crowding.

    M = size(PopObj,2);
    if K >= length(Last)
        Choose = true(1,length(Last));
        return;
    end

    if M <= 3
        [W,~] = UniformPoint(max(length(Last),K),M);
        Zmin  = min(PopObj(FrontNo==1,:),[],1);
        Zmax  = max(PopObj,[],1);
        NormObj = (PopObj - repmat(Zmin,size(PopObj,1),1))./repmat(max(Zmax-Zmin,1e-12),size(PopObj,1),1);
        NormObj = max(NormObj,0);
        W = W./repmat(sqrt(sum(W.^2,2)),1,M);
        Cosine = 1 - pdist2(NormObj(Last,:),W,'cosine');
        Cosine(isnan(Cosine)) = 0;
        Distance = repmat(sqrt(sum(NormObj(Last,:).^2,2)),1,size(W,1)).*sqrt(max(0,1-Cosine.^2));
        [d,associate] = min(Distance,[],2);
        rho = histcounts(associate,0.5:1:(size(W,1)+0.5));
        Choose = false(1,length(Last));
        while sum(Choose) < K
            remainRef = find(rho < inf);
            minRho = min(rho(remainRef));
            ref = remainRef(find(rho(remainRef)==minRho,1));
            cand = find(~Choose & associate' == ref);
            if isempty(cand)
                rho(ref) = inf;
            else
                if rho(ref) == 0
                    [~,p] = min(d(cand));
                else
                    p = randi(length(cand));
                end
                Choose(cand(p)) = true;
                rho(ref) = rho(ref) + 1;
            end
        end
    else
        CrowdDis = CrowdingDistance(PopObj,FrontNo);
        [~,rank] = sort(CrowdDis(Last),'descend');
        Choose = false(1,length(Last));
        Choose(rank(1:K)) = true;
    end
end
