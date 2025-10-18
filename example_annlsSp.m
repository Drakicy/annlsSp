rng(42);

func = @(x) 1 + sin(x).^2 + 4 * exp(-(x - 4).^2 / 2);

n = 100;
a = 0;
b = 10;
err = 0.1;

x = a + (b - a) * sort(rand(n, 1));
y = max(0, func(x) + err * randn(n, 1));

x_fine = linspace(x(1), x(end), 10 * n);

sp = annlsSp(x, y, Error=err);

figure;
hold on
plot(x, y, '.b');
plot(sp.knots, fnval(sp, sp.knots), '*r');

fnplt(sp)

func = @(x, y) 1 + sin(x + 2 * y).^2 + 4 * exp(-(x - 4).^2 / 2 - (y - 2).^2 / 3);

n = 100;
a = 0;
b = 10;
err = 0.1;

x = a + (b - a) * sort(rand(n, 1));
y = a + (b - a) * sort(rand(n, 1));

[X, Y] = ndgrid(x, y);

f = max(0, func(X, Y) + err * randn(n, n));

sp = annlsSp({x, y}, f, Error=err);

[X_knots, Y_knots] = ndgrid(sp.knots{:});

figure;
hold on
plot3(X, Y, f, '.b');
plot3(X_knots, Y_knots, fnval(sp, sp.knots), '*r');

fnplt(sp)

view(3)
