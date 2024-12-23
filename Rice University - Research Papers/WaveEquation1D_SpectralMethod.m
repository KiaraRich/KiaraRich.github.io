% % Spectral Method for 1D Wave Equation
%
% % This script implements the Fourier Spectral Method for
% % solving the 1D wave equation. The spatial domain is defined from 0 
% % to 2*pi with various discretizations to examine convergence. Initial 
% % conditions are based on a sinusoidal function, and solutions are 
% % obtained for the method over a fixed time period of 1 second.
% % Errors are calculated by comparing the numerical solutions to the 
% % exact analytical solution. Results, including error analysis and 
% % graphical representations of the solution, is displayed to evaluate 
% % the accuracy and efficiency of the method.

clc; clear;  % Clear command window and workspace, preparing for a fresh simulation run

% Initializing simulation parameters and error storage vectors
L = 2*pi;          % Domain length set to 2*pi, suitable for periodic boundary conditions
t_final = 1;       % Final simulation time in seconds
Nvec = [20, 40, 60, 80, 100];  % Different spatial discretizations to study convergence
m = length(Nvec);  % Total number of discretization scenarios
errs_spectral = zeros(m,1); % Vector to store errors from the spectral method

% Loop over each discretization scenario to compute solutions and errors
for j = 1:m
    N = Nvec(j);  % Number of spatial points in the current discretization
    h = L/N;      % Spatial step size, calculated as domain length divided by number of points
    c = 1;        % Wave speed, assumed unity for simplicity
    u0 = @(x) sin(x);  % Sinusoidal initial condition function
    dt = 0.0015;  % Time step size

    % Solve the wave equation using the Fourier Spectral method
    us = FoW(u0, L, N, c, dt, t_final);

    % Define and evaluate the exact solution at final time t_final
    u_true = @(x, t) sin(x - c*t);
    x = linspace(0, L, N+1);
    u_tval = u_true(x, t_final);

    % Calculate the L2 error norm for to measure accuracy
    errs_spectral(j) = sqrt(h * sum((us - u_tval).^2));
end

% Display a table of convergence
T = table(Nvec', errs_spectral, 'VariableNames', {'Spatial Discretization (N)', 'Spectral Method Error'});
disp('Table of Convergence =');
disp(T);

% Plotting the results
figure;
plot(x, us, 'm', x, u_tval, 'b--');
xlabel('Space');
ylabel('Displacement');
title('Wave Equation 1D: Spectral Method');
legend('Spectral Method', 'Exact Solution');

% Definition of the Fourier Spectral method function
function u = FoW(u0, L, N, c, dt, t_final)
% FoW Solves the 1D wave equation using the Fourier Spectral method
%
% This function uses the Fourier Spectral method to solve the 1D wave equation
% over a defined spatial domain from 0 to L with N spatial points, using a
% specified wave speed c, time step dt, and until final time t_final.
% The solution utilizes the Fast Fourier Transform (FFT) to move between
% the spatial and frequency domains.
%
% Inputs:
%   u0      - A function handle representing the initial condition.
%   L       - The length of the spatial domain.
%   N       - The number of spatial points in the domain.
%   c       - The wave speed.
%   dt      - The time step size.
%   t_final - The final time of the simulation.
%
% Outputs:
%   u       - The solution array at the final time, including periodic
%             boundary conditions.

    x = linspace(0, L, N+1);         % Generate spatial grid points
    u0val = u0(x(1:end-1));          % Evaluate initial condition at grid points, excluding the last point
    num_steps = round(t_final / dt); % Calculate the number of time steps based on final time and step size

    % Time-stepping loop using Fourier transform techniques
    for i = 1:num_steps
        u0_hat = fft(u0val);               % Compute Fourier coefficients of the current solution
        k = (-N/2:(N/2)-1) * (2*pi/L);     % Frequency vector for FFT, adjusted for MATLAB's FFT frequency ordering
        k = fftshift(k);                   % Shift zero frequency to center
        u_hat = exp(-1i*c*k*dt) .* u0_hat; % Apply wave equation in frequency domain
        u0val = real(ifft(u_hat));         % Inverse FFT to return to spatial domain
    end
    u = [u0val, u0val(1)];                 % Append first value at the end to ensure periodicity
end