classdef DMOFA < ALGORITHM
% <multi> <real/integer> <dynamic>
% Dynamic Multi-Objective Firefly Algorithm
% alpha --- 0.50 --- Random walk strength
% beta0 --- 1.50 --- Attractiveness at zero distance
% gamma --- 1.00 --- Light absorption coefficient
% zeta  --- 0.20 --- Ratio of random immigrants after each change
%
% This is a basic dynamic multi-objective firefly algorithm for myPlatemo.
% The main search operator is the standard firefly movement. Brightness is
% defined by Pareto rank and crowding distance, environmental changes are
% detected by re-evaluating sentinel solutions, and changes are handled by
% re-evaluation plus random immigrants.

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
            [alpha,beta0,gamma,zeta] = Algorithm.ParameterSet(0.50,1.50,1.00,0.20);
            Algorithm.save = sign(Algorithm.save)*inf;

            %% Initialization
            Population = Problem.Initialization();
            Population = DMOFA_EnvironmentalSelection(Population,Problem.N);
            AllPop     = [];

            %% Optimization
            while Algorithm.NotTerminated(Population)
                %% Detect environmental change
                if DMOFA_Changed(Problem,Population)
                    % Save the population before the change for dynamic metrics
                    AllPop = [AllPop,Population];

                    % Re-evaluate all current solutions in the new environment
                    Population = Problem.Evaluation(Population.decs);

                    % Basic dynamic response: random immigrants
                    nImm = max(1,ceil(zeta*Problem.N));
                    Imm  = Problem.Initialization(nImm);
                    Population = DMOFA_EnvironmentalSelection([Population,Imm],Problem.N);
                end

                %% Basic firefly search operator
                Offspring = DMOFA_OperatorFirefly(Problem,Population,alpha,beta0,gamma);

                %% Elitist environmental selection
                Population = DMOFA_EnvironmentalSelection([Population,Offspring],Problem.N);

                %% Return all populations for DF/FDA dynamic metric calculation
                if Problem.FE >= Problem.maxFE
                    Population = [AllPop,Population];
                    [~,rank]   = sort(Population.adds(zeros(length(Population),1)));
                    Population = Population(rank);
                end
            end
        end
    end
end

function changed = DMOFA_Changed(Problem,Population)
% Detect whether the dynamic problem has changed by re-evaluating 10% of
% representative solutions.

    N      = length(Population);
    nCheck = max(1,ceil(0.10*N));
    try
        [FrontNo,~] = NDSort(Population.objs,Population.cons,N);
    catch
        [FrontNo,~] = NDSort(Population.objs,N);
    end
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    CrowdDis(isnan(CrowdDis)) = 0;
    [~,rank] = sortrows([FrontNo(:),-CrowdDis(:)]);
    index    = rank(1:nCheck);

    RePop1  = Population(index);
    RePop2  = Problem.Evaluation(RePop1.decs);
    changed = ~isequal(RePop1.objs,RePop2.objs) || ~isequal(RePop1.cons,RePop2.cons);
end

function Offspring = DMOFA_OperatorFirefly(Problem,Population,alpha,beta0,gamma)
% Basic multi-objective firefly movement.
% Better fireflies are brighter. A firefly is attracted to one randomly
% selected brighter firefly; if no brighter firefly exists, it performs a
% random walk.

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

    % Rank is the primary brightness, crowding distance is used as tie-breaker.
    Brightness = -FrontNo(:) + 1e-3*CrowdDis(:);

    % Slowly reduce random walk strength to improve convergence.
    progress = min(1,Problem.FE/Problem.maxFE);
    alphaNow = alpha*(1-progress);

    OffDec = zeros(N,D);
    for i = 1 : N
        xi = Dec(i,:);
        brighter = find(Brightness > Brightness(i));
        brighter(brighter == i) = [];

        if isempty(brighter)
            beta = 0;
            xj   = xi;
        else
            j    = brighter(randi(length(brighter)));
            xj   = Dec(j,:);
            rij  = norm((xi-xj)./max(span,eps))/sqrt(D);
            beta = beta0*exp(-gamma*rij^2);
        end

        randomWalk  = alphaNow*(rand(1,D)-0.5).*span;
        OffDec(i,:) = xi + beta*(xj-xi) + randomWalk;
    end

    OffDec    = DMOFA_Repair(Problem,OffDec);
    Offspring = Problem.Evaluation(OffDec);
end

function Population = DMOFA_EnvironmentalSelection(Population,N)
% NSGA-II style environmental selection.

    if length(Population) <= N
        return;
    end
    try
        [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    catch
        [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    end
    Next = FrontNo < MaxFNo;
    Last = find(FrontNo == MaxFNo);
    K    = N - sum(Next);
    if K > 0
        CrowdDis = CrowdingDistance(Population.objs,FrontNo);
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:K))) = true;
    end
    Population = Population(Next);
end

function OffDec = DMOFA_Repair(Problem,OffDec)
% Bound and encoding repair.

    Lower  = repmat(Problem.lower,size(OffDec,1),1);
    Upper  = repmat(Problem.upper,size(OffDec,1),1);
    OffDec = min(max(OffDec,Lower),Upper);
    index  = ismember(Problem.encoding,2:4);
    OffDec(:,index) = round(OffDec(:,index));
end
