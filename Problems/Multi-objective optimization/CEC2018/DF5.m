classdef DF5 < PROBLEM
% <2018> <multi> <real> <large/none> <dynamic>
% Benchmark dynamic MOP from CEC2018 competition
% taut --- 10 --- Number of generations for static optimization
% nt   --- 10 --- Number of distinct steps

%------------------------------- Reference --------------------------------
% S. Jiang, S. Yang, X. Yao, K. C. Tan, M. Kaiser, and N. Krasnogor,
% Benchmark problems for CEC 2018 competition on dynamic multiobjective
% optimisation, in Proc. IEEE Congr. Evol. Comput., 2018, pp. 1-8.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        taut;
        nt;
    end
    methods
        function Setting(obj)
            [obj.taut,obj.nt] = obj.ParameterSet(10,10);
            obj.M = 2;
            if isempty(obj.D); obj.D = 10; end
            obj.lower    = zeros(1,obj.D);
            obj.upper    = ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
        end
        function Population = Evaluation(obj,varargin)
            PopDec     = obj.CalDec(varargin{1});
            PopObj     = obj.CalObj(PopDec);
            PopCon     = obj.CalCon(PopDec);
            Population = SOLUTION(PopDec,PopObj,PopCon,zeros(size(PopDec,1),1)+obj.FE);
            obj.FE     = obj.FE + length(Population);
        end
        function PopObj = CalObj(obj,PopDec)
            t = floor(obj.FE/obj.N/obj.taut)/obj.nt;
            H = 1./(1+exp(-5*(t-0.5)));
            G = sin(0.5*pi*t);
            g = 1 + sum((PopDec(:,2:end)-G).^2,2);
            PopObj(:,1) = PopDec(:,1);
            PopObj(:,2) = g.*H.*(1-PopDec(:,1)./g).^2 - H.*(0.2-PopDec(:,1)./g).^2;
        end
    end
end
