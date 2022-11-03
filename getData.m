% Generate the dataset for multi dimensional polynomial regression.

function [A, tau, b_0] = getData(nPerDim, ndim, dpoly, methodA, methodTarget, polyType)
    % A: The n by d data matrix.
    % tau: Raw leverage score of A. Not normalized.
    % b_0: The original target, or QoI. Not reshaped into 1D array.
    % nPerDim: The number of data points for each coordinate.
    % ndim: The number of dimensions.
    % dpoly: The polynomial degree of the regression.
    % methodA: How to pick data points in [-1, 1]^ndim space.
    %   Available methods: grid
    % methodTarget: How to generate the target values.
    %   Available methods: ODE
    % polyType: How to generate polynomial terms.
    %   Available types: Legendre, Chebyshev, None
    
    n = nPerDim^ndim;
    d = nchoosek(dpoly + ndim, ndim);
    
    if methodA == "grid"
        base = zeros(ndim, nPerDim); % The base data points without bias and polynomial terms.
        for dim = 1 : ndim
            base(dim, :) = linspace(-1, 1, nPerDim);
        end
        A = genAGrid(base);
    else
        error(methodA + " is not a valid method for generating the matrix A.");
    end

    tau = leverageScore(A);
    
    if methodTarget == "ODE"
        if ndim ~= 2
            error("ODE model is for 2D only.");
        else
            b_0 = genODE();
        end
    else
        error(methodTarget + " is not a valid method for generating the target b.");
    end

    function A = genAGrid(base)
        polyStraight = ones(ndim, nPerDim, dpoly + 1);
        polyStraight(:, :, 2) = base;
        for k = 3 : dpoly + 1
            if polyType == "Chebyshev"
                polyStraight(:, :, k) = 2 * polyStraight(:, :, k - 1) .* base - polyStraight(:, :, k - 2);
            elseif polyType == "Legendre"
                polyStraight(:, :, k) = (2 * k - 1) / k * polyStraight(:, :, k - 1) .* base - (k - 1) / k * polyStraight(:, :, k - 2);
            elseif polyType == "None"
                polyStraight(:, :, k) = polyStraight(:, :, k - 1) .* base;
            else
                error("Invalid polynomial type.")
            end
        end

        A = ones(n,d);
        P = zeros(n, ndim); % Permutation map for base data points.
        for i = 1 : ndim
            for j = 1 : nPerDim
                for k = 1 : nPerDim^(ndim - i)
                    P((j - 1) * nPerDim^(ndim - i) + k, i) = j;
                end
            end
            for l = 1 : nPerDim^(i - 1)
                P((l - 1) * nPerDim * nPerDim^(ndim - i) + 1 : l * nPerDim * nPerDim^(ndim - i), i) = P(1 : nPerDim * nPerDim^(ndim - i), i);
            end
        end
        C = nchoosek(1 : dpoly + ndim, ndim); % Permutation map for polynomials.
        for i = ndim : -1 : 2
            C(:, i) = C(:, i) - C(:, i - 1);
        end
        C = [sum(C, 2), C];
        [~, I] = sort(C, 1);
        C = C(I(:, 1), :);
        C = C(:, 2 : ndim + 1);

        for i = 1 : n
            for j = 1 : d
                for k = 1 : ndim
                    A(i, j) = A(i, j) * polyStraight(k, P(i, k), C(j, k));
                end
            end
        end
    end

    function tau = leverageScore(A) % Computing the leverage score.
        [U, ~, ~] = svd(A, 'econ');
        tau = sum(U.^2, 2);
    end

    function b_0 = genODE()
        % Example taken from https://www5.in.tum.de/lehre/vorlesungen/algo_uq/ss18/06_polynomial_chaos.pdf
        % Changing multiple variable: the forcing frequency w and spring constant k.
        c = 0.5;
        k = 2.0;
        f = 0.5;
        w = 0.8;
        p = [c, k, f, w];
        yinit = [0.5, 0];
        tmax = 20;
        kvals = 1 : 2 / (nPerDim - 1) : 3;
        wvals = 0 : 2 / (nPerDim - 1) : 2;
        b_0 = zeros(nPerDim, nPerDim);
        
        for i = 1 : nPerDim
            for j = 1 : nPerDim
                p(4) = wvals(i);
                p(2) = kvals(j);
                [~, y] = ode45(@(t, y) odemodel(t, y, p), [0, tmax], yinit);
                b_0(i, j) = max(y(:, 1));
            end
        end

        function deriv = odemodel(t, y, p)
            deriv = [y(2); p(3) * cos(p(4) * t) - p(2) * y(1) - p(1) * y(2)];
        end

    end
    
end