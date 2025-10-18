# Adaptive Nonnegative Least-Squares spline approximation

## Definition

Adaptive NonNegative Least-Squares (ANNLS) spline approximation is an algorithm that constructs a nonnegative spline function $S(x)$, approximating $f(x)$, based on a gridded data $(x_i, y_i)$. The set of knots is constructed automatically on each iteration of the algorithm, based on the error distribution. 

The theoretical basis of the algorithm and some applications are described in following articles:
1. **WIP**
2. Jingu Kim, Yunlong He, and Haesun Park. Algorithms for Nonnegative Matrix and Tensor Factorizations: A Unified View Based on Block Coordinate Descent Framework, Journal of Global Optimization, [link](http://dx.doi.org/10.1007/s10898-013-0035-4)

Refer to the first publication if the algorithm is used in a scientific work.

## Represantation

The algorithm is represented as a function in MATLAB (see examples).

The grid has to be represented through grid vectors:
```
x = {x_1, x_2, x_3, ...}; %cell array of vectors
%in a univariate case, x can be scalar
```

$y = f(x)$ has to be set as an array that is compatiable with the grid:
```
y = ...; %multidimensional array
```

The spline function can be constructed as follows:
```
[sp, err_sp] = annlsSp( ...
    x, ...
    y ...
  );
```
Output is a MATLAB spline object and an error of the approximation.

The algorithm halts after becoming stable. The stability is assessed by evaluating the change of the function integral on the approximation interval:
```
tol = ...; %positive scalar less than 1 (default 1 / 10^3)
```

To prevent early halting due to an inefficient step, an upper bound of spline approximation error can be introduced:
```
err = ...; %positive scalar (default Inf)
```

To describe a more complex behavior, the order of the spline can be changed:
```
k = ...; %scalar (unified) or vector (for each axis)
```

Complete algorithm initialization can be represented as:
```
[sp, err_sp] = annlsSp( ...
    x, ...
    y, ...
    Tol=tol, ...
    Error=err, ...
    Order=k ...
  );
```
