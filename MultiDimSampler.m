% A class of sampling for multi-dimensional data.

classdef MultiDimSampler
        
    properties
        A               % The n by d data matrix.
        uniformProb     % Equal inclusion probability for all data points. Sum up to 1.
        leverageProb    % Inclusion probability proportional to leverage score. Sum up to 1.
        dist            % n by n matrix. Each row stores the index of data points in order of l2 distance.
        n               % The number of data points.
        ndim            % The number of dimensions.
        dpoly           % The polynomial degree of the regression.
        d               % The number of features.
        ndiv            % The number of groups that the coordSampling split the data points into in each iteration.
    end
    
    methods
        
        % Initialization.
        function obj = MultiDimSampler(A, levScore, ndim, dpoly, ndiv)
            obj.A = A;
            obj.n = size(A, 1);
            obj.d = size(A, 2);
            obj.ndim = ndim;
            obj.dpoly = dpoly;
            obj.leverageProb = levScore / sum(levScore);
            obj.uniformProb = ones(obj.n, 1) / obj.n;
            obj.dist = distance(obj);
            obj.ndiv = ndiv;

            function dist = distance(obj) 
                dist = zeros(obj.n);
                for i = 1 : obj.n - 1
                    for j = i + 1 : obj.n
                        dist(i, j) = sum((A(i, :) - A(j, :)).^2);
                        dist(j, i) = dist(i, j);
                    end
                end
                index = 1 : obj.n;
                for i = 1 : obj.n
                    [~, I] = sort(dist(i, :));
                    dist(i, :) = index(I);
                end
            end
        end

        function obj = setNumDiv(ndiv)
            obj.ndiv = ndiv;
        end
        
        % Main function.
        function [index, prob] = sampling(obj, s, methodSample, methodProb)
            if ismember(methodProb, ["uniform", "leverage"])
                if methodSample == "bernoulli"
                    [index, prob] = bernoulliSampling(obj, s, methodProb);
                elseif methodSample == "withReplacement"
                    [index, prob] = withReplacementSampling(obj, s, methodProb);
                elseif methodSample == "pivotalDistance"
                    [index, prob] = pivotalDistanceSampling(obj, s, methodProb);
                elseif methodSample == "pivotalCoordwise"
                    [index, prob] = pivotalCoordSampling(obj, s, methodProb, "coordwise");
                elseif methodSample == "pivotalPCA"
                    [index, prob] = pivotalCoordSampling(obj, s, methodProb, "PCA");
                else
                    error("No sampling method is available for " + methodSample + " with " + methodProb + " as inclusion probability.");
                end
            else 
                error("No sampling method is available for " + methodSample + " with " + methodProb + " as inclusion probability.");
            end
        end

        function prob = setProb(obj, s, methodProb)
            if methodProb == "uniform"
                prob = obj.uniformProb * s;
            elseif methodProb == "leverage"
                prob = obj.leverageProb * s;
            end
            if max(prob) > 1
                % No implementation for deterministic choosing.
                error("The number of sample is too much against the number of data points generated.");
            end
        end
        
        % Bernoulli Sampling: For each data point, decide picking it or not from the corresponding probability.
        function [index, prob] = bernoulliSampling(obj, s, methodProb)
            prob = setProb(obj, s, methodProb);
            index = [];
            for i = 1 : obj.n
                if rand() < prob(i)
                    index = [index, i];
                end
            end
            prob = prob(index);
        end
        
        % Sampling with replacement: Pick one item following the probability, and repeat it until getting enough samples.
        function [index, prob] = withReplacementSampling(obj, s, methodProb)
            prob = setProb(obj, s, methodProb);
            probCum = zeros(obj.n, 1);
            probCum(1) = prob(1);
            for i = 2 : obj.n
                probCum(i) = probCum(i - 1) + prob(i);
            end

            index = zeros(s, 1);
            choice = sort(rand(s, 1) * s);
            i = 1;
            j = 1;
            while (i <= s)
                if choice(i) < probCum(j)
                    index(i) = j;
                    i = i + 1;
                else
                    j = j + 1;
                end
            end
            prob = prob(index);
        end
        
        % Pivotal Sampling based on the distance between a pair of points.
        % https://www.jstor.org/stable/23270453#metadata_info_tab_contents
        function [index, prob] = pivotalDistanceSampling(obj, s, methodProb)
            prob = setProb(obj, s, methodProb);
            unfinished = 1 : obj.n;
            isfinished = zeros(obj.n, 1);
            pi = prob;
            for counter = obj.n : -1 : 2
                iPtr = randi(counter);
                i = unfinished(iPtr);
                for k = 2 : obj.n
                    if isfinished(obj.dist(unfinished(iPtr), k)) == 0
                        j = obj.dist(unfinished(iPtr), k);
                        jPtr = binarySearch(unfinished, j);
                        break;
                    end
                end

                if pi(i) + pi(j) < 1
                    if rand * (pi(i) + pi(j)) < pi(j)
                        pi(j) = pi(j) + pi(i);
                        pi(i) = 0;
                        unfinished(iPtr) = [];
                        isfinished(i) = 1;
                    else
                        pi(i) = pi(i) + pi(j);
                        pi(j) = 0;
                        unfinished(jPtr) = [];
                        isfinished(j) = 1;
                    end
                else 
                    if rand * (2 - pi(i) - pi(j)) < 1 - pi(j)
                        pi(j) = pi(j) + pi(i) - 1;
                        pi(i) = 1;
                        unfinished(iPtr) = [];
                        isfinished(i) = 1;
                    else
                        pi(i) = pi(i) + pi(j) - 1;
                        pi(j) = 1;
                        unfinished(jPtr) = [];
                        isfinished(j) = 1;
                    end
                end
            end

            index = zeros(s, 1);
            iPtr = 1;
            for i = 1 : obj.n
                if pi(i) > 0.5
                    index(iPtr) = i;
                    iPtr = iPtr + 1;
                end
            end
            prob = prob(index);

            function loc = binarySearch(arr, x)
                loc = binarysearch(arr, x, 1, length(arr));
                function loc = binarysearch(arr, x, l, r)
                    mid = ceil((l + r) / 2);
                    if arr(mid) == x
                        loc = mid;
                    elseif arr(mid) > x
                        loc = binarysearch(arr, x, l, mid - 1);
                    else 
                        loc = binarysearch(arr, x, mid + 1, r);
                    end
                end
            end
        end
        
        % Pivotal Sampling with partitioning, budgeting, and shifting.
        % https://epubs.siam.org/doi/10.1137/21M1422513
        function [index, prob] = pivotalCoordSampling(obj, s, methodProb, type)
            prob = setProb(obj, s, methodProb);
            q = prob;
            partition((1 : obj.n)', 0);
            index = zeros(s, 1);
            iPtr = 1;
            for i = 1 : obj.n
                if q(i) > 0.5
                    index(iPtr) = i;
                    iPtr = iPtr + 1;
                end
            end
            prob = prob(index);

            function partition(member, count)
                if length(member) == 1
                    return;
                end
                if type == "coordwise" 
                    % Pick one axis in order, partition on the selected axis values.
                    sortCoord = mod(count, obj.ndim) + 1;
                    mat = [member, obj.A(member, sortCoord)];
                elseif type == "PCA" % Partition on the maximum variance direction.
                    C = pca(obj.A(member, :));
                    mat = [member, obj.A(member, :) * C(:, 1)];
                end
                [~, I] = sort(mat, 1);
                memberSorted = member(I(:, 2));
                
                ngroup = min(obj.ndiv, length(member));
                groupIndex = zeros(ngroup, 2);
                groupIndex(:, 2) = round((1 : ngroup) / ngroup * length(member));
                groupIndex(:, 1) = [1, groupIndex(1 : ngroup - 1, 2)' + 1];

                a = zeros(ngroup, 1);
                t = zeros(ngroup, 1);
                g = zeros(ngroup, 1);
                for j = 1 : ngroup
                    a(j) = sum(q(memberSorted(groupIndex(j, 1) : groupIndex(j, 2))));
                    t(j) = a(j) - floor(a(j));
                    g(j) = floor(a(j));
                end
                g = g + budgeting(t);
                
                for j = 1 : ngroup
                    if g(j) > 0
                        if g(j) > a(j)
                            for k = groupIndex(j, 1) : groupIndex(j, 2)
                                y = min(1, q(memberSorted(k)) / t(j));
                                a(j) = a(j) + y - q(memberSorted(k));
                                q(memberSorted(k)) = y;
                                if a(j) >= g(j)
                                    q(memberSorted(k)) = y + g(j) - a(j);
                                    break
                                end
                            end
                        else
                            for k = groupIndex(j, 1) : groupIndex(j, 2)
                                y = max(0, (q(memberSorted(k)) - t(j)) / (1 - t(j)));
                                a(j) = a(j) + y - q(memberSorted(k));
                                q(memberSorted(k)) = y;
                                if a(j) <= g(j)
                                    q(memberSorted(k)) = y + g(j) - a(j);
                                    break
                                end
                            end
                        end
                        partition(memberSorted(groupIndex(j, 1) : groupIndex(j, 2)), count + 1);
                    else
                        q(memberSorted(groupIndex(j, 1) : groupIndex(j, 2))) = zeros(groupIndex(j, 2) - groupIndex(j, 1) + 1, 1);
                    end
                end
            end

            function budget = budgeting(t)
                budget = zeros(length(t), 1);
                g = sum(t);
                b = 0;
                l = 0;
                f = 1;
                h = 0;
                for j = 1 : round(g)
                    probList = b;
                    es = f;
                    while es < length(t) && probList(length(probList)) + t(es) < 1
                        probList = [probList, probList(length(probList)) + t(es)];
                        es = es + 1;
                    end
                    random = rand() * probList(length(probList));
                    for pl_ptr = 1 : length(probList)
                        if random < probList(pl_ptr)
                            if pl_ptr == 1
                                h = l;
                            else
                                h = f + pl_ptr - 2;
                            end
                            break
                        end
                    end
                    a = 1 - sum(probList);
                    if es > length(t)
                        b = 0;
                    else
                        b = t(es) - a;
                    end
                    if rand() < 1 - a / (1 - b)
                        budget(h) = 1;
                        l = es;
                    else
                        budget(es) = 1;
                        l = h;
                    end
                    f = es + 1;
                end
            end

        end

    end % method

end % class