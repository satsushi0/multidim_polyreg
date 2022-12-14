% Multi-dimensional polynomial regression with sampling.

nPerDim = 100;                      % Number of points to generate for each coordinate.
ndim = 2;                           % Degree of dimensionality.
dpoly = 5;                          % Degree of polynomial.
n = nPerDim^ndim;                   % Total number of points.
d = nchoosek(dpoly + ndim, ndim);   % The number of features.

[A, tau, b_0] = getData(nPerDim, ndim, dpoly, 'grid', 'ODE', 'Legendre');

sampleSize = [30, 35, 40, 45, 50, 60, 70, 80];
ntrial = 100;

sampleMethods = ["bernoulli", "withReplacement", "pivotalDistance", "pivotalCoordwise", "pivotalPCA"];
probMethods = ["uniform", "leverage"];

errors = zeros(length(sampleMethods) * length(probMethods), length(sampleSize), ntrial);

mds = MultiDimSampler(A, tau, ndim, dpoly, 2);
b = reshape(b_0, n, 1);
b_norm = mean(b.^2);

for i = 1 : length(sampleSize)
    disp("Iteration " + num2str(i) + " / " + num2str(length(sampleSize)) + " .....");
    s = sampleSize(i);
    for j = 1 : length(probMethods)
        pm = probMethods(j);
        for k = 1 : length(sampleMethods)
            sm = sampleMethods(k);
            for t = 1 : ntrial
                [index, prob] = mds.sampling(s, sm, pm);
                A_tilde = A(index, :) ./ (prob.^(1 / 2));
                b_tilde = b(index) ./ (prob.^(1 / 2));
                X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde;
                errors((j - 1) * length(sampleMethods) + k, i, t) = mean((A * X_tilde - b).^2) / b_norm;
            end
        end
    end
end

medError = median(errors, 3);

% Plot the result.
ls = containers.Map(1 : length(sampleMethods), [":", ":", "--", "-", "-"]);
color = containers.Map(1 : length(sampleMethods), ["#1d104a", "#2e8a6a", "#580023", "#bf4616", "#ffc000"]);

figure();
hold on;
for m = 1 : length(sampleMethods)
    plot(sampleSize, medError(m, :), 'LineWidth', 1, 'LineStyle', ls(m), 'Color', color(m));
end
set(gca, 'YScale', 'log');
ylim([0.005, 0.1]);
title("Uniform Probability", 'FontSize', 12);
xlabel("# samples");
ylabel("Median Normalized Error");
legend(sampleMethods, 'FontSize', 10);
grid on;
hold off;

figure();
hold on;
for m = 1 + length(sampleMethods) : 2 * length(sampleMethods);
    plot(sampleSize, medError(m, :), 'LineWidth', 3, 'LineStyle', ls(m - length(sampleMethods)), 'Color', color(m - length(sampleMethods)));
end
set(gca, 'YScale', 'log');
ylim([0.005, 0.1]);
title("Leverage Score Probability", 'FontSize', 12);
xlabel("# samples");
ylabel("Median Normalized Error");
legend(sampleMethods, 'FontSize', 10);
grid on;
hold off;

figure();
hold on;
plot(sampleSize, medError(3, :), 'LineWidth', 1, 'LineStyle', ls(3), 'Color', color(3));
plot(sampleSize, medError(4, :), 'LineWidth', 1, 'LineStyle', ls(4), 'Color', color(4));
plot(sampleSize, medError(5, :), 'LineWidth', 1, 'LineStyle', ls(5), 'Color', color(5));
plot(sampleSize, medError(8, :), 'LineWidth', 3, 'LineStyle', ls(3), 'Color', color(3));
plot(sampleSize, medError(9, :), 'LineWidth', 3, 'LineStyle', ls(4), 'Color', color(4));
plot(sampleSize, medError(10, :), 'LineWidth', 3, 'LineStyle', ls(5), 'Color', color(5));
set(gca, 'YScale', 'log');
ylim([0.005, 0.1]);
title("Spatial Method Comparison", 'FontSize', 12);
xlabel("# samples");
ylabel("Median Normalized Error");
legend(["(u)pivotalDistance", "(u)pivotalCoordwise", "(u)pivotalPCA", "(ls)pivotalDistance", "(ls)pivotalCoordwise", "(ls)pivotalPCA"], 'FontSize', 10);
grid on;
hold off;


