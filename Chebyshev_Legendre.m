% Visualization to understand the behaviors of Chebyshev and Legendre Polynomials.

n = 50;     % Number of data points to generate for each coordinate.
dpoly = 5;  % The degree of polynomial.

% Generate the data, compute the parameter X using all the data.
[A_leg, tau_leg, b_leg] = getData(n, 2, dpoly, 'grid', 'ODE', 'Legendre');
b_leg = reshape(b_leg, n^2, 1);
X_leg = (A_leg' * A_leg) \ A_leg' * b_leg;
[A_che, tau_che, b_che] = getData(n, 2, dpoly, 'grid', 'ODE', 'Chebyshev');
b_che = reshape(b_che, n^2, 1);
X_che = (A_che' * A_che) \ A_che' * b_che;
[A, tau, b] = getData(n, 2, dpoly, 'grid', 'ODE', 'None');
b = reshape(b, n^2, 1);
X = (A' * A) \ A' * b;

figure();
hold on;
for i = 1 : size(A_leg, 2)
    scatter(A_leg(:, 2), A_leg(:, i));
end
set(gca, 'YScale', 'log');
ylim([1e-9, 1]);
xlabel("Base");
ylabel("Legendre Polynomials (positive values only)");
title("Data Points by Legendre Polynomials with Polynomial Degree " + num2str(dpoly), 'FontSize', 14);
hold off;

figure();
hold on;
for i = 1 : size(A_che, 2)
    scatter(A_che(:, 2), A_che(:, i));
end
set(gca, 'YScale', 'log');
ylim([1e-9, 1]);
xlabel("Base");
ylabel("Chebyshev Polynomials (positive values only)");
title("Data Points by Chebyshev Polynomials with Polynomial Degree " + num2str(dpoly), 'FontSize', 14);
hold off;

figure();
hold on;
for i = 1 : size(A, 2)
    scatter(A(:, 2), A(:, i));
end
set(gca, 'YScale', 'log');
ylim([1e-9, 1]);
xlabel("Base");
ylabel("Simple Polynomials (positive values only)");
title("Data Points by Simple Polynomials with Polynomial Degree " + num2str(dpoly), 'FontSize', 14);
hold off;

% Plot the model parameters. Using X.^exp for y-axis to make plot easier to see.
exp = 1 / 3;
X_leg(X_leg > 0) = X_leg(X_leg > 0).^(exp);
X_leg(X_leg < 0) = -(-X_leg(X_leg < 0)).^(exp);
X_che(X_che > 0) = X_che(X_che > 0).^(exp);
X_che(X_che < 0) = -(-X_che(X_che < 0)).^(exp);
X(X > 0) = X(X > 0).^(exp);
X(X < 0) = -(-X(X < 0)).^(exp);
figure();
hold on;
scatter(1 : length(X_leg), X_leg, 70, 'filled', 'square');
scatter(1 : length(X_che), X_che, 70, 'filled', 'diamond');
scatter(1 : length(X), X, 70, 'filled', 'o');
xlabel("low  -- degree of polynomial (correspond to columns of A) --   high");
ylabel("The Parameter in X to the power of " + num2str(round(exp, 3)));
grid on;
legend(["Legendre", "Chebyshev", "Simple"]);
title("The Model Parameters", 'FontSize', 14);
hold off;

% Simulation of leverage sampling with different polynomials.
sampleSize = [30, 35, 40, 45, 50, 60, 70, 80];
ntrial = 100;

errors_leg = zeros(length(sampleSize), ntrial);
mds = MultiDimSampler(A_leg, tau_leg, 2, dpoly, 2);
b_norm_leg = mean(b_leg.^2);
for smp = 1 : length(sampleSize)
    s = sampleSize(smp);
    for t = 1 : ntrial
        [index, prob] = mds.sampling(s, "pivotalCoordwise", "leverage");
        A_tilde = A_leg(index, :) ./ (prob.^(1 / 2));
        b_tilde = b_leg(index) ./ (prob.^(1 / 2));
        X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde;
        errors_leg(smp, t) = mean((A_leg * X_tilde - b).^2) / b_norm_leg;
    end
end

errors_che = zeros(length(sampleSize), ntrial);
mds = MultiDimSampler(A_che, tau_che, 2, dpoly, 2);
b_norm_che = mean(b_che.^2);
for smp = 1 : length(sampleSize)
    s = sampleSize(smp);
    for t = 1 : ntrial
        [index, prob] = mds.sampling(s, "pivotalCoordwise", "leverage");
        A_tilde = A_che(index, :) ./ (prob.^(1 / 2));
        b_tilde = b_che(index) ./ (prob.^(1 / 2));
        X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde;
        errors_che(smp, t) = mean((A_che * X_tilde - b).^2) / b_norm_che;
    end
end

errors = zeros(length(sampleSize), ntrial);
mds = MultiDimSampler(A, tau, 2, dpoly, 2);
b_norm = mean(b.^2);
for smp = 1 : length(sampleSize)
    s = sampleSize(smp);
    for t = 1 : ntrial
        [index, prob] = mds.sampling(s, "pivotalCoordwise", "leverage");
        A_tilde = A(index, :) ./ (prob.^(1 / 2));
        b_tilde = b(index) ./ (prob.^(1 / 2));
        X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde;
        errors(smp, t) = mean((A * X_tilde - b).^2) / b_norm;
    end
end

figure();
hold on;
plot(sampleSize, median(errors_leg, 2), "LineWidth", 3, "LineStyle", "-");
plot(sampleSize, median(errors_che, 2), "LineWidth", 3, "LineStyle", "--");
plot(sampleSize, median(errors, 2), "LineWidth", 3, "LineStyle", ":");
set(gca, 'YScale', 'log');
ylim([0.005, 0.1]);
title("Coordwise Sampling with Different Polynomials", 'FontSize', 12);
xlabel("# samples");
ylabel("Median Normalized Error");
legend(["Legendre", "Chebyshev", "Simple"], 'FontSize', 10);
grid on;

