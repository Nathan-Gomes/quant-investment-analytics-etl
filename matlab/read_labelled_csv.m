function [labels, values, header] = read_labelled_csv(path)
% READ_LABELLED_CSV  Read a CSV whose first column is a label and the rest numeric.
%
%   Uses only core file I/O, so the same code runs in MATLAB and in GNU Octave.
%   readtable would be shorter and is not available in Octave, which would mean
%   this cross-check could only be run by someone with a MATLAB licence — a poor
%   property for a script whose purpose is letting other people verify a result.

    handle = fopen(path, 'r');
    if handle < 0
        error('Could not open %s', path);
    end
    cleaner = onCleanup(@() fclose(handle));

    header_line = strtrim(fgetl(handle));
    header = strsplit(header_line, ',');

    labels = {};
    values = [];
    while true
        line = fgetl(handle);
        if ~ischar(line)
            break;
        end
        line = strtrim(line);
        if isempty(line)
            continue;
        end
        parts = strsplit(line, ',');
        labels{end + 1, 1} = parts{1};                                  %#ok<AGROW>
        numbers = zeros(1, numel(parts) - 1);
        for i = 2:numel(parts)
            numbers(i - 1) = str2double(parts{i});
        end
        values(end + 1, :) = numbers;                                   %#ok<AGROW>
    end
end
