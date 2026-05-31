function changed = FAMM_Changed(Problem,Population)
% 环境变化检测：重评估10%个体
    RePop1  = Population(randperm(end,ceil(end/10)));
    RePop2  = Problem.Evaluation(RePop1.decs);
    changed = ~isequal(RePop1.objs,RePop2.objs) || ~isequal(RePop1.cons,RePop2.cons);
end
