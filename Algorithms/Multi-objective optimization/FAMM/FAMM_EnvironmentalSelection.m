function [Population,Archive,Fitness] = FAMM_EnvironmentalSelection(Population,N)
% SPEA2风格环境选择

    Fitness = FAMM_CalFitness(Population.objs);

    Next    = Fitness < 1;
    Archive = Population(Next);
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
        Archive = Population(Next);
    end
    Population = Population(Next);
    Fitness    = Fitness(Next);
end

function Del = Truncation(PopObj,K)
% 基于欧氏距离的截断
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
