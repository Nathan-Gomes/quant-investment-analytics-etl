function weights = risk_parity_portfolio(sigma, maximum_weight)
% RISK_PARITY_PORTFOLIO  Equal risk contribution weights.
%
%   Minimising 0.5 * w' * sigma * w - sum(b .* log(w)) over w > 0 has the
%   first-order condition sigma*w = b ./ w, which says every holding supplies
%   risk in proportion to its budget b. Written instead as "minimise the spread
%   of risk contributions" the same problem is not convex, and a local search on
%   it degrades as the universe grows.
%
%   Fixing every other weight leaves a quadratic in w(i) with one positive root,
%   so cyclical coordinate descent solves it in closed form with no solver at
%   all. Spinu (2013); Maillard, Roncalli and Teiletche (2010).

    if nargin < 2
        maximum_weight = 1.0;
    end
    n = size(sigma, 1);
    budget = repmat(1 / n, n, 1);
    weights = repmat(1 / sqrt(n), n, 1);

    for sweep = 1:3000
        movement = 0;
        for i = 1:n
            cross = sigma(i, :) * weights - sigma(i, i) * weights(i);
            a = sigma(i, i);
            if a <= 0
                continue;
            end
            updated = (-cross + sqrt(cross^2 + 4 * a * budget(i))) / (2 * a);
            movement = max(movement, abs(updated - weights(i)));
            weights(i) = updated;
        end
        if movement < 1e-14
            break;
        end
    end

    weights = weights / sum(weights);
    if max(weights) > maximum_weight + 1e-9
        % Exact equal contribution and a binding cap cannot both hold. The cap
        % is the mandate, so the solution is projected onto it.
        weights = minimum_variance_portfolio(eye(n), maximum_weight) * 0 + ...
                  project_simplex(weights, maximum_weight);
    end
end


function projected = project_simplex(vector, maximum_weight)
    low = min(vector) - maximum_weight - 1;
    high = max(vector) + 1;
    projected = vector;
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
