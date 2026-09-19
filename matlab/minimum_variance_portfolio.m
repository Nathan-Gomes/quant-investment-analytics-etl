function weights = minimum_variance_portfolio(sigma, maximum_weight)
% MINIMUM_VARIANCE_PORTFOLIO  Long-only, fully invested minimum-variance weights.
%
%   weights = MINIMUM_VARIANCE_PORTFOLIO(sigma, maximum_weight) minimises
%   w' * sigma * w subject to sum(w) == 1 and 0 <= w <= maximum_weight.
%
%   This is an independent implementation rather than a second call to the same
%   library, for two reasons: a cross-check that shares a solver with the thing
%   it checks tests less than it appears to, and a reviewer without an
%   Optimization Toolbox licence can still run it. It also runs unmodified in
%   GNU Octave, which is how it is verified in continuous integration.
%
%   The method is accelerated projected gradient. The objective is convex and
%   the feasible set is a box intersected with the budget constraint, so the
%   iteration converges to the global optimum; projecting onto that set leaves a
%   single scalar unknown, which bisection finds.
%
%   Agreement with the Python pipeline is asserted by verify_against_python.m.

    if nargin < 2
        maximum_weight = 1.0;
    end
    n = size(sigma, 1);
    if n * maximum_weight < 1 - 1e-12
        error('A cap of %.2f cannot be spread across %d assets.', maximum_weight, n);
    end

    % The gradient of w'*sigma*w is 2*sigma*w, so the objective is Lipschitz
    % with constant 2*lambda_max and this step size is the largest that is safe.
    step = 1 / (2 * max(abs(eig(sigma))));

    current = project_to_budget(repmat(1 / n, n, 1), maximum_weight);
    momentum = current;
    theta = 1;

    for iteration = 1:20000
        gradient = 2 * (sigma * momentum);
        candidate = project_to_budget(momentum - step * gradient, maximum_weight);

        movement = max(abs(candidate - current));
        next_theta = (1 + sqrt(1 + 4 * theta^2)) / 2;
        momentum = candidate + ((theta - 1) / next_theta) * (candidate - current);

        current = candidate;
        theta = next_theta;
        if movement < 1e-13 && iteration > 50
            break;
        end
    end
    weights = current;
end


function projected = project_to_budget(vector, maximum_weight)
% Nearest point in {w : sum(w) == 1, 0 <= w <= maximum_weight}. Clamping at a
% shifted level is monotone in the shift, so exactly one shift makes the result
% sum to one, and bisection finds it.
    low = min(vector) - maximum_weight - 1;
    high = max(vector) + 1;
    projected = min(max(vector, 0), maximum_weight);
    for step = 1:200
        shift = (low + high) / 2;
        projected = min(max(vector - shift, 0), maximum_weight);
        total = sum(projected);
        if abs(total - 1) < 1e-14
            return;
        end
        if total > 1
            low = shift;
        else
            high = shift;
        end
    end
end
