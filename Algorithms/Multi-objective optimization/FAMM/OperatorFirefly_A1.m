function Offspring = OperatorFirefly_A1(Problem,Population,Fitness,alphaExp,betaExp,gammaExp,alphaConv,betaConv,gammaConv,eliteRate,mem_factor,pConv)
% Adaptive hybrid firefly operator
% 每个个体按 pConv 自适应选择：
% - convergence mode：温和 C3 精英吸引
% - exploration mode：原始 FAMM 多父本吸引 + Levy 飞行

    N    = length(Population);
    D    = Problem.D;
    lb   = Problem.lower;
    ub   = Problem.upper;
    span = max(ub-lb,1e-12);

    [FrontNo,~] = NDSort(Population.objs,Population.cons,N);
    CrowdDis    = CrowdingDistance(Population.objs,FrontNo);

    nElite   = max(2,ceil(eliteRate*N));
    [~,rank] = sortrows([FrontNo(:),Fitness(:),-CrowdDis(:)]);
    EliteIdx = rank(1:nElite);
    EliteDec = Population(EliteIdx).decs;
    EliteC   = mean(EliteDec,1);

    OffDec = zeros(N,D);
    for i = 1:N
        xi = Population(i).dec;

        if rand < pConv
            %% Convergence mode: moderated C3 operator
            betterElite = EliteIdx(FrontNo(EliteIdx) < FrontNo(i));
            if isempty(betterElite)
                j = EliteIdx(randi(length(EliteIdx)));
            else
                j = betterElite(randi(length(betterElite)));
            end
            xj = Population(j).dec;

            r    = norm((xi-xj)./span) / sqrt(D);
            beta = betaConv * exp(-gammaConv*r^2);
            beta = min(beta,1.35 - 0.20*mem_factor);

            if FrontNo(i) == 1
                contract = 0.075*mem_factor*(EliteC-xi);
            else
                contract = 0.035*mem_factor*(EliteC-xi);
            end

            noise = alphaConv * randn(1,D).*span*0.025;
            OffDec(i,:) = xi + beta*(xj-xi) + contract + noise;
        else
            %% Exploration/tracking mode: original FAMM-like operator
            brighter_idx = find(FrontNo < FrontNo(i));
            if ~isempty(brighter_idx)
                k = min(5,length(brighter_idx));
                selected = brighter_idx(randperm(length(brighter_idx),k));
                delta = zeros(1,D);
                weight_sum = 0;
                for s = 1:k
                    xj = Population(selected(s)).dec;
                    r  = norm((xi-xj)./span) / sqrt(D);
                    beta_ij = betaExp * exp(-gammaExp*r^2);
                    delta = delta + beta_ij*(xj-xi);
                    weight_sum = weight_sum + beta_ij;
                end
                if weight_sum > 0
                    delta = delta ./ weight_sum;
                end
                levy_step = LevyFlightA1(1,D).*span*0.045;
                OffDec(i,:) = xi + delta + alphaExp*levy_step;
            else
                % 第一前沿个体保留轻微随机游走以维持动态跟踪能力。
                OffDec(i,:) = xi + alphaExp*randn(1,D).*span*0.085;
            end
        end

        OffDec(i,:) = max(min(OffDec(i,:),ub),lb);
    end

    Offspring = Problem.Evaluation(OffDec);
end

function L = LevyFlightA1(n,d)
% Levy flight step, beta = 1.5
    beta = 1.5;
    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u = randn(n,d)*sigma;
    v = randn(n,d);
    L = u./(abs(v).^(1/beta));
end
