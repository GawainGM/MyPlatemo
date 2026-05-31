function [Population,Archive,Centroid] = FAMM_ChangeResponse(Problem,Population,Archive,Centroid,mem_charge)
% 环境变化响应：保留一半旧解重评估，质心预测生成新解

    remain = ~Truncation(Population.objs,ceil(length(Population)/2));
    Population(remain) = Problem.Evaluation(Population(remain).decs);

    if ~isobject(Archive) || ~isa(Archive,'SOLUTION') || numel(Archive) == 0
        Ct = mean(Population.decs,1);
    else
        Ct = mean(Archive.decs,1);
    end
    St = norm(Ct - Centroid);

    Archive = Population(remain).best;
    if ~isobject(Archive) || ~isa(Archive,'SOLUTION') || numel(Archive) == 0
        CA = mean(Population(remain).decs,1);
    else
        CA = mean(Archive.decs,1);
    end
    CR = mean(Population(remain).decs,1);
    X  = Population(~remain).decs;
    if size(CA,2) == size(CR,2) && size(CA,2) == size(X,2)
        X = X + repmat(St.*(CA-CR)./norm(CA-CR),size(X,1),1) + randn(size(X)).*St/2/sqrt(size(X,2));
    else
        X = Problem.CalDec(rand(size(X)));
    end
    Population(~remain) = Problem.Evaluation(X);

    Archive  = Population.best;
    Centroid = Ct;
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
