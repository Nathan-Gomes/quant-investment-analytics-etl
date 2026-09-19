% VERIFY_AGAINST_PYTHON  Independent MATLAB check of the Strata optimizer.
%
% Strata solves minimum variance and risk parity three ways: with cvxpy in the
% Python service, with projected gradient and coordinate descent in the browser
% build, and here in MATLAB. Three implementations of the same two programs are
% only worth maintaining if they are checked against each other, so this
% compares against the Python answer and fails loudly on disagreement.
%
%   python -m tools.export_for_matlab
%   octave --no-gui matlab/verify_against_python.m
%   matlab -batch "run('matlab/verify_against_python.m')"
%
% No toolbox required, and it runs unmodified in GNU Octave so a reviewer
% without a MATLAB licence can still check the result.

here = fileparts(mfilename('fullpath'));
if isempty(here)
    here = pwd;
end
root = fileparts(here);
addpath(here);

folder = fullfile(root, 'outputs', 'matlab');
covariance_file = fullfile(folder, 'covariance.csv');
weights_file = fullfile(folder, 'python_weights.csv');
settings_file = fullfile(folder, 'settings.csv');
if ~exist(covariance_file, 'file')
    error('Export the inputs first: python -m tools.export_for_matlab');
end

[~, settings_values, settings_header] = read_labelled_csv(settings_file);
maximum_weight = settings_values(1, strcmp(settings_header(2:end), 'maximum_weight'));

[tickers, sigma] = read_labelled_csv(covariance_file);   % already annualised on export
[~, python_values, weights_header] = read_labelled_csv(weights_file);
python_all = [python_values(:, strcmp(weights_header(2:end), 'minimum_variance')), ...
              python_values(:, strcmp(weights_header(2:end), 'risk_parity'))];

matlab_all = [minimum_variance_portfolio(sigma, maximum_weight), ...
              risk_parity_portfolio(sigma, maximum_weight)];

problems = {'minimum_variance'; 'risk_parity'};
tolerance = 1e-4;
gaps = zeros(numel(problems), 1);
passed = true;

fprintf('\nStrata: independent MATLAB cross-check\n');
fprintf('================================================================\n');
for p = 1:numel(problems)
    fprintf('\n%s (cap %.0f%%)\n', strrep(problems{p}, '_', ' '), maximum_weight * 100);
    fprintf('%-10s %14s %14s %12s\n', 'ticker', 'python', 'matlab', 'difference');
    for i = 1:numel(tickers)
        fprintf('%-10s %14.6f %14.6f %12.2e\n', tickers{i}, python_all(i, p), ...
                matlab_all(i, p), abs(python_all(i, p) - matlab_all(i, p)));
    end
    gaps(p) = max(abs(python_all(:, p) - matlab_all(:, p)));
    fprintf('largest difference %.3e (tolerance %.0e)\n', gaps(p), tolerance);

    assert(abs(sum(matlab_all(:, p)) - 1) < 1e-10, 'MATLAB weights are not fully invested.');
    assert(all(matlab_all(:, p) > -1e-12), 'MATLAB weights include a short position.');
    assert(max(matlab_all(:, p)) <= maximum_weight + 1e-9, 'MATLAB weights breach the cap.');
    passed = passed && (gaps(p) < tolerance);
end

% Risk parity has a defining property, so check that rather than only the weights.
parity = matlab_all(:, 2);
contributions = parity .* (sigma * parity) / sqrt(parity' * sigma * parity);
fprintf('\nrisk contributions differ by %.3e across holdings\n', max(contributions) - min(contributions));

fprintf('\n================================================================\n');
if passed
    fprintf('RESULT: PASS - the implementations agree.\n\n');
else
    fprintf('RESULT: FAIL - the implementations disagree beyond tolerance.\n\n');
end

write_csv(fullfile(folder, 'matlab_cross_check.csv'), ...
    {'problem', 'max_weight_difference', 'passed'}, problems, ...
    [gaps, double(gaps < tolerance)]);

if ~passed
    error('Strata MATLAB cross-check failed.');
end
