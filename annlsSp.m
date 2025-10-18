function [sp, err_sp] = annlsSp(x, f, opt)
    %% annlsSp: adaptive nonnegative least-squares spline approximation
    %
    % INPUT
    %
    %   x: cell array of grid vectors {x_1, x_2, ..., x_n}
    %
    %   f: n-dimensional data array
    %
    %   (optional) err: upper bound of spline approximation error,
    %                   positive scalar
    %
    %   (optional) tol: relative spline integral error,
    %                   positive scalar less than 1
    %
    %   (optional) k: spline order,
    %                 scalar (unified) or vector (for each axis)
    %
    % OUTPUT
    %
    %   sp: spline object

    arguments
        x (:,1)
        f {mustBeNonnegative}
        opt.Error (1,1) {mustBePositive} = Inf
        opt.Tol (1,1) {mustBePositive, mustBeLessThan(opt.Tol, 1)} = 1e-3
        opt.Order (:,1) {mustBeInteger, mustBeGreaterThanOrEqual(opt.Order, 0)} = 3
    end

    %% Set basic parameters and perform argument validation

    % Set grid dimensionality
    if iscell(x)
        x_dim = length(x);
    else
        x_dim = 1;

        x = {x};
        f = f(:);
    end

    % Set grid size
    x_size = zeros(x_dim, 1);

    for i = 1:x_dim
        % Check whether grid vectors are vectors
        assert( ...
                isvector(x{i}), ...
                'Data:notVector', ...
                "Elements of 'x' must be vectors." ...
            );

        % Check whether grid vectors are real
        assert( ...
                all(isreal(x{i})), ...
                'Data:notReal', ...
                "Elements of 'x' must be real." ...
            );

        % Check whether grid vectors are sorted
        assert( ...
                issorted(x{i}), ...
                'Data:notSorted', ...
                "Elements of 'x' must be sorted." ...
            );

        x{i} = x{i}(:);
        x_size(i) = length(x{i});

        % Check whether data size agrees with grid size
        assert( ...
            isequal(size(f, i), x_size(i)), ...
            'Size:incompatible', ...
            "Dimensionality of 'f' must agree with dimensionality of 'x'." ...
        );
    end

    % Set spline order
    if isscalar(opt.Order)
        k = opt.Order * ones(x_dim, 1);
    else
        % Check whether spline order size agrees with grid dimensionality
        assert( ...
            length(opt.Order) == x_dim, ...
            'Size:incompatible', ...
            "Size of 'k' must agree with dimensionality of 'x'." ...
        );

        k = opt.Order;
    end

    %% Perform approximation

    % Set initial knots indexes
    knots_ind = cellfun(@(c) [1 length(c)]', x, UniformOutput=false);
    
    % Set initial spline integral
    int = 0;

    % Perform knot selection
    while true  
        % Find nonnegative least-squares spline approximation
        sp = nnlsSp(x, f, knots_ind, k);

        % Calculate residuals and error
        f_sp = fnval_nd(sp, x);
        res2 = (f - f_sp).^2;
        err_sp = sqrt(mean(res2, 'all'));
        
        % Calculate spline integral
        int_ = int;
        int = fnval_nd(fnder(sp, -1 * ones(x_dim, 1)), cellfun(@(c) c(end), x, UniformOutput=false));

        % Check whether stop criteria is satisfied
        if ((abs(int - int_) <= opt.Tol * int_) && (err_sp < opt.Error)) || all(k .* cellfun(@length, knots_ind) > x_size)
            return;
        end

        % Add new knots
        knots_ind = newKnots(knots_ind, res2, k);
    end
end

function sp = nnlsSp(x, f, knots_ind, k)
    x_dim = length(x);
    f_size = size(f);

    knots = cell(x_dim, 1);

    for i = 1:x_dim
        knots{i} = x{i}( ...
            [
                knots_ind{i}(1) * ones(k(i), 1)
                knots_ind{i} 
                knots_ind{i}(end) * ones(k(i), 1)
            ]);

        B = spcol(knots{i}, k(i) + 1, x{i}, 'sparse', 'noderiv');

        f = nnlsm_blockpivot(B, reshape(f, f_size(1), []));

        f_size(1) = size(f, 1);
        f = reshape(f, f_size);
        
        if x_dim > 1
            f = permute(f, [2:x_dim 1]);
            f_size = f_size([2:x_dim 1]);
        end
    end

    sp = spmak_nd(knots, f);
end

function knots_ind = newKnots(knots_ind, res2, k)
    knots_dim = length(knots_ind);

    % Find an intervals with maximum cumulative residual
    res2_max_ind = cell(knots_dim, 1);
    res2_max = 0;

    for i = 1:knots_dim
        if k(i) * length(knots_ind{i}) >= size(res2, i)
            continue;
        end

        res2_i = sum(res2, setdiff(1:knots_dim, i));
        res2_i = res2_i(:);

        for j = 1:length(knots_ind{i})-1
            if knots_ind{i}(j) + 1 == knots_ind{i}(j+1)
                continue;
            end

            res2_i_bin = sum(res2_i(knots_ind{i}(j):knots_ind{i}(j+1)));

            if (j ~= 1)
                res2_i_bin = res2_i_bin - res2_i(knots_ind{i}(j)) / 2;
            end

            if (j + 1 ~= length(knots_ind{i}))
                res2_i_bin = res2_i_bin - res2_i(knots_ind{i}(j+1)) / 2;
            end

            if res2_i_bin == res2_max
                res2_max_ind{i} = [res2_max_ind{i} j];
            elseif res2_i_bin > res2_max
                res2_max = res2_i_bin;

                res2_max_ind = cell(knots_dim, 1);
                res2_max_ind{i} = j;
            end
        end
    end

    % Add new knots
    for i = 1:knots_dim
        if isempty(res2_max_ind{i})
            continue;
        end

        res2_i = sum(res2, setdiff(1:knots_dim, i));
        res2_i = res2_i(:);

        cnt = 0;

        for j = res2_max_ind{i}
            bin_res2 = res2_i(knots_ind{i}(j+cnt));

            if j ~= 1
                bin_res2 = bin_res2 / 2;
            end

            bin_res2 = bin_res2 + res2_i(knots_ind{i}(j+cnt)+1);

            pos = 1;

            while (2 * bin_res2 < res2_max) && (knots_ind{i}(j+cnt) + pos + 1 < knots_ind{i}(j+cnt+1))
                pos = pos + 1;

                bin_res2 = bin_res2 + res2_i(knots_ind{i}(j+cnt)+pos);
            end

            knots_ind{i} = ...
                [
                    knots_ind{i}(1:j+cnt)
                    knots_ind{i}(j+cnt) + pos
                    knots_ind{i}(j+cnt+1:end)
                ];

            cnt = cnt + 1;
        end
    end
end

% Syntax fix

function sp = fnval_nd(sp, x)
    if isscalar(x)
        sp = fnval(sp, x{1});
    else
        sp = fnval(sp, x);
    end
end

function sp = spmak_nd(knots, coefs)
    if isscalar(knots)
        sp = spmak(knots{1}, coefs(:)');
    else
        sp = spmak(knots, coefs);
    end
end