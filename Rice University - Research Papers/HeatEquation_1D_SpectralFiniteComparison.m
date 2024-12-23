% % Spectral Methods for PDEs - 1D Heat Equation
% 
% % This script implements and compares the Fourier Spectral Method and  
% % Finite Difference methods for solving the 1D heat equation.
% % The domain is defined from 0 to 2*pi with various discretizations to
% % examine convergence. Initial conditions are based on the sinusoidal
% % function, and solutions are obtained for both methods over a fixed time
% % period. Errors are calculated by comparing numerical solutions to the
% % exact analytical solution. Results, including error analysis and
% % graphical representations of the solutions, are displayed to evaluate the
% % accuracy and efficiency of each method.

clc; clear;  % Clear command window and workspace

% Initializing simulation parameters and error storage vectors
L = 2*pi;          % Define the domain length as 2*pi
t_final = 1;       % Set the final time for the simulation
% Nvec = [20, 40, 60, 80, 100];  % Array of spatial discretization points
Nvec = [50,100,150,200,250]; % Larger Nvec for higher accuracy, longer computation 
m = length(Nvec);  % Number of different discretizations
errs = zeros(m,1); % Initialize vector to store errors for spectral method
errf = zeros(m,1); % Initialize vector to store errors for finite difference method

% Loop over each discretization scenario
for j = 1:m
    N = Nvec(j);         % Current number of spatial points
    h = L/(Nvec(j)+1);   % Calculate the spatial step size h based on the domain length L and the number of intervals Nvec(j)
    alpha = 1;           % Diffusion coefficient hardcoded to 1
    u0 = @(x) sin(x);    % Initial condition as a function handle
    dt = 0.5 * h^2;      % Time step size

    % Solve using the Fourier Spectral method
    us = FoD(u0, L, N, alpha, dt, t_final);

    % Solve using the Finite Difference method
    uf = finite1(u0, L, N, alpha, dt, t_final);

    % Define the exact solution and evaluate it at t_final
    u_true = @(x, t) sin(x)*exp(-t);
    x = linspace(0, L, N+1);
    u_tval = u_true(x, t_final);

    % Calculate the L2 error norm for both methods
    errs(j) = sqrt(h * sum((us - u_tval).^2));
    errf(j) = sqrt(h * sum((uf - u_tval).^2));
end

% Display the table of convergence in 1D
T = table(Nvec', errf, errs);
disp('Table of Convergence =')
disp(T)

% Plot #1 - Heat Equation 1D: Fourier Spectral Method
figure(1);
plot(x, us, 'r');
xlabel('Space');
ylabel('Temperature');
title('Heat Equation 1D: Fourier Spectral Method')

% Plot #2 - Heat Equation 1D: Finite Difference Method
figure(2);
plot(x, uf, 'g');
xlabel('Space');
ylabel('Temperature');
title('Heat Equation 1D: Finite Difference Method')

% Plot #3 - Heat Equation 1D: True Solution
figure(3);
plot(x, u_tval, 'b');
xlabel('Space');
ylabel('Temperature');
title('Heat Equation 1D: True Solution')

% Plot #4 - Heat Equation 1D: Errors
figure(4);
plot(Nvec, errf, 'b-o', Nvec, errs, 'r-*');
xlabel('N');
ylabel('Error');
title('Convergence History');
legend('Finite', 'Spectral');

% Plot #5 - Heat Equation 1D: All Methods Comparison
figure(5);
plot(x, u_tval, 'r', x, us, 'b', x, uf, 'g');
xlabel('Space');
ylabel('Temperature');
title('Heat Equation Solutions');
legend('Exact', 'Spectral', 'Finite');

% Fourier Spectral method for 1D heat equation
function u = FoD(u0, L, N, alpha, dt, t_final)
% Solves the heat equation using the Fourier Spectral method, which involves
% calculating the Fourier transform of the initial conditions, applying the heat equation
% in the Fourier domain, and transforming back to get the solution in the spatial domain
%
% Inputs:
%   u0      - Function handle for the initial condition.
%   L       - Length of the spatial domain.
%   N       - Number of spatial grid points.
%   alpha   - Diffusion coefficient.
%   dt      - Time step size.
%   t_final - Final time for the simulation.
%
% Outputs:
%   u       - Approximate solution to the heat equation at final time.

    x = linspace(0, L, N+1);         % Create spatial grid along x-axis
    u0val = u0(x(1:end-1));          % Evaluate initial condition on spatial grid
    num_steps = round(t_final / dt); % Calculate number of time steps

    % Time-stepping loop
    for i = 1:num_steps
        % Compute Fourier coefficients of u_0 (transfer to frequency space)
        u0_hat = fft(u0val);   % use fft for 1D

        % Compute Fourier frequencies (frequency vector k)
        k = (-N/2:(N/2)-1);    % Create an array of frequencies for FFT, adjusted for MATLAB's FFT frequency ordering
        k = fftshift(k);       % Shift zero frequency to center, suitable for symmetric operations in frequency space
        ik = -(k.*k);          % Compute negative squared frequencies, representing the Laplacian in the frequency domain        
        ii = find(ik ~= 0);    % Find indices of non-zero frequencies to avoid division by zero in next step
        ikinverse = ik;        % Copy the ik array to prepare for inverting non-zero elements
        ikinverse(ii) = 1./ik(ii); % Compute the reciprocal of non-zero elements of ik for the spectral method       
        Lap = ikinverse .* u0_hat; % Multiply by Fourier coefficients to apply the Laplacian in the frequency domain

        % Solve for Fourier coefficients at next time step
        u_hat = u0_hat + alpha * dt * Lap;

        % Inverse Fourier transform to get solution at continued time step
        u0val = real(ifft(u_hat)); % Convert back to spatial domain, use ifft for 1D
    end
    u = [u0val, u0val(1)];         % Append the first value to the end to close the domain
end

% Finite Difference Method for 1D heat equation
function uf = finite1(u0, L, N, alpha, dt, t_final)
% Solves the heat equation using a finite difference approach, discretizing
% the spatial domain with a central difference scheme and applying an explicit
% time-stepping method
%
% Inputs:
%   u0      - Function handle for the initial condition.
%   L       - Length of the spatial domain.
%   N       - Number of spatial grid points.
%   alpha   - Diffusion coefficient.
%   dt      - Time step size.
%   t_final - Final time for the simulation.
%
% Outputs:
%   uf      - Approximate solution to the heat equation at final time.

    % Create a spatial grid and calculate the spatial step size dx
    x = linspace(0, L, N+1);
    dx = L/(N+1);

    % Evaluate the initial condition at the interior grid points
    uf = u0(x(2:end)');  

    % Initialize the tridiagonal matrix A representing the Laplacian operator
    % using a finite difference approximation (central difference scheme)
    e = ones(N, 1);
    A = 1/(dx*dx) * spdiags([e -2*e e], [-1 0 1], N, N);

    % Apply periodic boundary conditions by modifying the corners of matrix A
    A(1, end) = 1/(dx*dx);
    A(end, 1) = 1/(dx*dx);

    % Calculate the number of time steps needed based on the final time and the time step size
    num_steps = round(t_final/dt);

    % Time-stepping loop: advance solution through time using the explicit Euler method
    for i = 1:num_steps
        Un = A * uf;               % Calculate the next time step solution using the matrix-vector multiplication
        uf = uf + alpha * dt * Un; % Update the solution vector with the time-stepped solution
    end
    % Wrap around the periodic boundary by appending the last element to the beginning
    uf = [uf(end), uf'];
end