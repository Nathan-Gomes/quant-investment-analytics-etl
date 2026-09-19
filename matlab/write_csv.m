function write_csv(path, header, labels, values)
% WRITE_CSV  Write a label column followed by numeric columns.
%   Core I/O only, so it runs in MATLAB and in GNU Octave.

    handle = fopen(path, 'w');
    if handle < 0
        error('Could not write %s', path);
    end
    cleaner = onCleanup(@() fclose(handle));

    fprintf(handle, '%s', header{1});
    for i = 2:numel(header)
        fprintf(handle, ',%s', header{i});
    end
    fprintf(handle, '\n');

    for row = 1:numel(labels)
        fprintf(handle, '%s', labels{row});
        fprintf(handle, ',%.10g', values(row, :));
        fprintf(handle, '\n');
    end
end
