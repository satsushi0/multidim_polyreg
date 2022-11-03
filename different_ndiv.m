% Simulation on coordwise sampling with different ndiv values.
% ndiv: The number of groups the algorithm partitions the data points into in each iteration.

nPerDim = 100;                      % Number of points to generate for each coordinate.
ndim = 2;                           % Degree of dimensionality.
dpoly = 5;                          % Degree of polynomial.
n = nPerDim^ndim;                   % Total number of points.
d = nchoosek(dpoly + ndim, ndim);   % The number of features.

[A, tau, b_0] = getData(nPerDim, ndim, dpoly, 'grid', 'ODE', 'Legendre');

sampleSize = [30, 35, 40, 45, 50, 60, 70, 80];
ntrial = 100;
ndivs = [2, 3, 4, 5];

errors = zeros(length(ndivs), length(sampleSize), ntrial);

b = reshape(b_0, n, 1);
b_norm = mean(b.^2);

for nd = 1 : length(ndivs)
    mds = MultiDimSampler(A, tau, ndim, dpoly, ndivs(nd));
    for i = 1 : length(sampleSize)
        s = sampleSize(i);
        for t = 1 : ntrial
            [index, prob] = mds.sampling(s, "pivotalCoordwise", "leverage");
            A_tilde = A(index, :) ./ (prob.^(1 / 2));
            b_tilde = b(index) ./ (prob.^(1 / 2));
            X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde;
            errors(nd, i, t) = mean((A * X_tilde - b).^2) / b_norm;
        end
    end
end

medError = median(errors, 3);

% Plot the result.

ls = containers.Map(1 : length(ndivs), ["-.", ":", "--", "-"]);

figure();
hold on;
for nd = 1 : length(ndivs)
    plot(sampleSize, medError(nd, :), 'LineWidth', 3, 'LineStyle', ls(nd));
end
set(gca, 'YScale', 'log');
ylim([0.005, 0.1]);
title("Coordwise with leverage with different ndiv", 'FontSize', 12);
xlabel("# samples");
ylabel("Median Normalized Error");
legend(num2str(ndivs'), 'FontSize', 10);
grid on;
hold off;
