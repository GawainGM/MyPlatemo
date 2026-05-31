function Offspring = OperatorFirefly(Problem,Population,alpha,beta0,gamma)
% 萤火虫位置更新算子
% 基于Pareto前沿等级吸引：排名更优（FrontNo更小）的个体吸引较差个体
% 多父本加权吸引 + Lévy飞行随机扰动

    N = length(Population);
    D = Problem.D;
    lb = Problem.lower;
    ub = Problem.upper;

    % 非支配排序：获取每个个体的前沿编号
    [FrontNo,~] = NDSort(Population.objs,inf);

    OffDec = zeros(N,D);
    for i = 1:N
        xi = Population(i).dec;

        % 找到比当前个体Pareto更优的个体
        brighter_idx = find(FrontNo < FrontNo(i));

        if ~isempty(brighter_idx)
            % 最多选取5个更优个体作为吸引源（模拟忆阻器交叉阵列并行计算）
            k = min(5, length(brighter_idx));
            selected = brighter_idx(randperm(length(brighter_idx), k));

            % 加权平均位移：Σ β_ij * (x_j - x_i) / Σ β_ij
            delta = zeros(1, D);
            weight_sum = 0;
            for s = 1:k
                xj = Population(selected(s)).dec;
                r = norm(xi - xj);
                beta_ij = beta0 * exp(-gamma * r^2);
                delta = delta + beta_ij * (xj - xi);
                weight_sum = weight_sum + beta_ij;
            end
            if weight_sum > 0
                delta = delta / weight_sum;
            end

            % Lévy飞行随机扰动
            levy_step = LevyFlight(1, D) .* (ub - lb) * 0.05;
            OffDec(i,:) = xi + delta + alpha * levy_step;
        else
            % 当前个体已在第一前沿，仅做随机扰动
            OffDec(i,:) = xi + alpha * randn(1, D) .* (ub - lb) * 0.1;
        end

        % 边界约束
        OffDec(i,:) = max(min(OffDec(i,:), ub), lb);
    end

    Offspring = Problem.Evaluation(OffDec);
end

function L = LevyFlight(n, d)
% 生成Lévy飞行步长（β=1.5的Mantegna算法）
    beta = 1.5;
    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u = randn(n,d) * sigma;
    v = randn(n,d);
    L = u ./ (abs(v).^(1/beta));
end
