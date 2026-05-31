clear; clc;
addpath(genpath(pwd));

runs  = 30;
N     = 100;
M     = 3;
D     = 10;
maxFE = 10000;

IGDValues = nan(runs,1);
HVValues  = nan(runs,1);
Runtime   = nan(runs,1);

for r = 1 : runs
    rng(r,'twister');
    Problem   = DF14('N',N,'M',M,'D',D,'maxFE',maxFE);
    Algorithm = FAMM_A3('save',1,'outputFcn',@(Algorithm,Problem)[]);
    Algorithm.Solve(Problem);

    Population   = Algorithm.result{end,2};
    IGDValues(r) = Problem.CalMetric('IGD',Population);
    HVValues(r)  = Problem.CalMetric('HV',Population);
    Runtime(r)   = Algorithm.CalMetric('runtime');

    fprintf('Run %02d/%02d: IGD = %.10g, HV = %.10g, runtime = %.2fs, FE = %d\n', ...
        r,runs,IGDValues(r),HVValues(r),Runtime(r),Problem.FE);
end

fprintf('\nFAMM on DF, %d runs, N=%d, M=%d, D=%d, maxFE=%d\n',runs,N,M,D,maxFE);
fprintf('Mean IGD = %.10g\n',mean(IGDValues,'omitnan'));
fprintf('Std  IGD = %.10g\n',std(IGDValues,'omitnan'));
fprintf('Mean HV  = %.10g\n',mean(HVValues,'omitnan'));
fprintf('Std  HV  = %.10g\n',std(HVValues,'omitnan'));
fprintf('Mean runtime = %.2fs\n',mean(Runtime,'omitnan'));
fprintf('Std  runtime = %.2fs\n',std(Runtime,'omitnan'));

%save('Data/FAMM_DF_summary.mat','IGDValues','HVValues','Runtime','runs','N','M','D','maxFE');
