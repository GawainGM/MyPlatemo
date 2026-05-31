function [Population,Archive,Fitness] = FAMM_EnvironmentalSelection_C4(Population,N)
% C4 环境选择：约束优先 + 归一化目标 SPEA2 适应度 + 截断

    PopObj = Population.objs;
    PopCon = Population.cons;

    % 约束违反度。无约束问题中 Population.cons 为空时，全部视为可行。
    if isempty(PopCon)
        CV = zeros(length(Population),1);
    else
        CV = sum(max(0,PopCon),2);
    end
    Feasible = CV <= 0;

    % 对目标归一化后计算 SPEA2 fitness，避免不同目标量纲主导选择。
    PopObjN  = NormalizeObjectives(PopObj);
    Fitness  = inf(1,length(Population));

    if any(Feasible)
        FitF = FAMM_CalFitness(PopObjN(Feasible,:));
        Fitness(Feasible) = FitF;
    end

    Next = false(1,length(Population));
    FeaIdx = find(Feasible);
    InfIdx = find(~Feasible);

    if length(FeaIdx) >= N
        % 可行解足够时，只在可行解中做 SPEA2 选择与密度截断。
        NextF = Fitness(FeaIdx) < 1;
        if sum(NextF) < N
            [~,Rank] = sort(Fitness(FeaIdx));
            NextF(Rank(1:N)) = true;
        elseif sum(NextF) > N
            Del  = TruncationC4(PopObjN(FeaIdx(NextF),:),sum(NextF)-N);
            Temp = find(NextF);
            NextF(Temp(Del)) = false;
        end
        Next(FeaIdx(NextF)) = true;
    else
        % 可行解不足时，全部保留，并用约束违反度最小的不可行解补齐。
        Next(FeaIdx) = true;
        [~,RankCV] = sort(CV(InfIdx));
        Need = N - length(FeaIdx);
        Next(InfIdx(RankCV(1:Need))) = true;

        % 给不可行个体一个可排序的 fitness，便于主算法更新 mem_charge。
        if ~isempty(InfIdx)
            CVN = CV(InfIdx)' ./ max(max(CV(InfIdx)),1e-12);
            Fitness(InfIdx) = 1 + CVN;
        end
    end

    Archive    = Population(Next & Feasible');
    Population = Population(Next);
    Fitness    = Fitness(Next);

    % 极端情况下没有可行 archive，则用当前选择种群中最优个体兜底。
    if isempty(Archive)
        Archive = Population.best;
    end
end

function PopObjN = NormalizeObjectives(PopObj)
% 目标归一化到 [0,1] 附近，处理退化目标和 NaN/Inf。
    PopObj(~isfinite(PopObj)) = 1e30;
    fmin = min(PopObj,[],1);
    fmax = max(PopObj,[],1);
    PopObjN = (PopObj - fmin) ./ max(fmax - fmin,1e-12);
end

function Del = TruncationC4(PopObj,K)
% 基于归一化目标空间欧氏距离的截断。
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
