% % Spectral Methods for PDEs - 2D Heat Equation
% 
% % This script implements and compares the Fourier Spectral Method and  
% % Finite Difference methods for solving the 2D heat equation.
% % The domain is defined from 0 to 2*pi on both x and y axes with various 
% % discretizations to examine convergence. Initial conditions are based on
% % a sinusoidal function, and solutions are obtained for both methods over 
% % a fixed time period. Errors are calculated by comparing numerical solutions
% % to the exact analytical solution. Results, including error analysis and
% % graphical representations of the solutions, are displayed to evaluate the
% % accuracy and efficiency of each method.

clc; clear;    % Clear command window and workspace

% Initializing simulation parameters and error storage vectors
L = 2*pi;      % Define the domain length as 2*pi
t_final = 1;   % Set the final time for the simulation

% Choose Nvec size for different accuracy levels
% Nvec = [20,40,60,80,100];      % Nvec size for testing accuracy
Nvec = [50,100,150,200,250];     % Larger Nvec for higher accuracy, longer computation 

m = length(Nvec);              % Number of different discretizations
errs = zeros(m,1);             % Initialize vector to store errors for spectral method
errf = zeros(m,1);             % Initialize vector to store errors for finite difference method

% Loop over each discretization scenario
for j = 1:m
    h = L/(Nvec(j)+1);          % Calculate the spatial step size h based on the domain length L and the number of intervals Nvec(j)
    Nx = Nvec(j);               % Make the number of grid points along the x-axis to the current value in Nvec
    Ny = Nvec(j);               % Make the number of grid points along the y-axis to the current value in Nvec
    alpha = 1;                  % Set the diffusion coefficient to be constant
    u0 = @(x,y) sin(x).*sin(y); % Define the initial condition function for the 2D heat equation
    dt = 0.25 * h^2;            % Set the time step size dt, scaling with the square of the spatial grid size to maintain stability
    
    % Solve using the Fourier Spectral method
    U = Fo2D(u0,L,Nx,Ny,alpha, dt, t_final);
    
    % Solve using the Finite Difference method
    UF = finite_diff(L,Nx,Ny,alpha, dt, t_final);
    
    % Define the exact solution and evaluate it at t_final
    u_true = @(x,y, t) sin(x).*sin(y)*exp(-2*t);  % Define a function for the exact solution of the heat equation
    x = linspace(0,L,Nx+1);        % Create a vector of linearly spaced points along the x-axis from 0 to L, including both boundaries
    y = linspace(0,L,Ny+1);        % Create a vector of linearly spaced points along the y-axis from 0 to L, including both boundaries
    [X,Y] = meshgrid(x,y);         % Generate a 2D grid of coordinates from the x and y vectors, where X is the matrix of x-coordinates and Y is the matrix of y-coordinates
    u_tval = u_true(X,Y,t_final);  % Evaluate the exact solution at the final time t_final over the meshgrid, storing the results in u_tval
    
    % Calculate the L2 error norm for both methods
    errs(j) = sqrt(h*h*sum((U-u_tval).^2, 'all'));
    errf(j) = sqrt(h*h*sum((UF-u_tval).^2, 'all'));
end

% Display the table of convergence in 2D
k = 1:m;
T = table(k',errf,errs);
disp('Table of Convergence =')
disp(T)
 
% Plot #1 - Heat Equation 2D: Fourier Spectral Method
figure (1);
surf(X,Y,U); 
xlabel('X');
ylabel('Y');
zlabel('Temperature');
title('Heat Equation 2D: Fourier Spectral Method')

% Plot #2 - Heat Equation 2D: Finite Difference Method
figure (2);
surf(X,Y,UF); 
xlabel('X');
ylabel('Y');
zlabel('Temperature');
title('Heat Equation 2D: Finite Difference Method')

% Plot #3 - Heat Equation 2D: True solution
figure (3);
surf(X,Y,u_tval);
xlabel('X');
ylabel('Y');
zlabel('Temperature');
title('Heat Equation 2D: True solution')

% Plot #4 - Heat Equation 2D: Errors
figure (4);
plot(Nvec,errf,'b-o',Nvec,errs,'r-*');
xlabel('N');
ylabel('Error');
title('Convergence History')
legend('Finite','Spectral')

% Fourier Spectral method for 2D heat equation
function u = Fo2D(u0,L,Nx,Ny,alpha, dt, t_final)
% This function computes the solution to the 2D heat equation over a square
% domain using the Fourier spectral method, employing periodic boundary conditions.
%
% Inputs:
%   u0      - Initial condition function handle (u0(x, y))
%   L       - Domain length (0 to L in both x and y directions)
%   Nx      - Number of grid points along the x-axis
%   Ny      - Number of grid points along the y-axis
%   alpha   - Diffusion coefficient
%   dt      - Time step size
%   t_final - Final time for simulation
%
% Outputs:
%   u       - Approximate solution to the heat equation at t_final as a 2D array
    
    x = linspace(0, L, Nx+1);      % Create spatial grid along x-axis
    y = linspace(0, L, Ny+1);      % Create spatial grid along x-axis
    [X, Y] = meshgrid(x(1:end-1), y(1:end-1)); % Create meshgrid along x and y axes (exclude last point for periodicity)
    u0val = u0(X,Y);               % Evaluate initial conditions on the grid
    num_steps = round(t_final/dt); % Calculate the number of time steps to reach t_final

    % Time stepping loop
    for i = 1:num_steps
        % Compute Fourier coefficients of the initial conditions (move to frequency space)
        u0_hat = fft2(u0val);  % use fft2 now for 2D problem
 
        % Frequency vector
        kx = (-Nx/2:(Nx/2)-1); % Generate a vector of frequencies for the Fourier transform along the x-axis
        ky = (-Ny/2:(Ny/2)-1); % Generate a vector of frequencies for the Fourier transform along the y-axis

        % Center the zero frequency component
        kx = fftshift(kx); % Shift the zero frequency to the center of the spectrum for the x-axis
        ky = fftshift(ky); % Shift the zero frequency to the center of the spectrum for the y-axis
 
        % Square and negate frequencies for Laplacian
        ikx = -(kx.*kx); % Compute the square of each frequency component along the x-axis, negate it to prepare for the Laplacian computation in the frequency domain
        iky = -(ky.*ky); % Compute the square of each frequency component along the y-axis, negate it to prepare for the Laplacian computation in the frequency domain
 
        % Find indices of non-zero frequency components to avoid division by zero
        iix = find(ikx ~= 0);
        iiy = find(iky ~= 0);

        % Inverse of non-zero frequency components for spectral method Laplacian calculation
        ikxinverse = ikx;               % Copy the modified frequency values for x to a new variable for inversion
        ikyinverse = iky;               % Copy the modified frequency values for y to a new variable for inversion
        ikxinverse(iix) = 1./ikx(iix);  % Calculate and store the inverse of the non-zero x-frequency components to apply the spectral Laplacian
        ikyinverse(iiy) = 1./iky(iiy);  % Calculate and store the inverse of the non-zero y-frequency components to apply the spectral Laplacian
     
        % Meshgrid to calculate 2D Laplacian in frequency domain
        [K1,K2] = meshgrid(ikxinverse,ikyinverse);  % Create a 2D grid of inverse frequency values
        Lap  =  (K1 + K2)  .* u0_hat;               % Calculate the Laplacian in the frequency domain

        % Update Fourier coefficients for the next time step
        u_hat = u0_hat + alpha * dt * Lap;
 
        % Inverse Fourier transform to return to spatial domain at continued time step
        u0val = real(ifft2(u_hat));     % use ifft2 now for 2D problem
     end
 
     % Append the first column and first row to mimic periodic boundary conditions
     u = u0val;        % Update u0 for the continued time step
     u = [u u(:,1)];
     u = [u; u(1,:)];
end

% Finite Difference Method for 2D Heat Equation
function U = finite_diff(L,Nx,Ny,alpha, dt, t_final)
% This function calculates the solution of the 2D heat equation over a square
% domain using the finite difference method with an explicit time-stepping scheme
%
% Inputs:
%   L       - Length of the domain (assumed square domain, same for x and y)
%   Nx      - Number of grid points along the x-axis
%   Ny      - Number of grid points along the y-axis
%   alpha   - Thermal diffusivity constant
%   dt      - Time step size
%   t_final - Final time for the simulation
%
% Outputs:
%   U       - Solution of the heat equation at the final time t_final, represented as a 2D grid

    % Create spatial grids along x and y axes
    x = linspace(0, L, Nx+1);                    % Linearly spaced vector for x-axis
    y = linspace(0, L, Ny+1);                    % Linearly spaced vector for y-axis
    dx = L/(Nx+1);                               % Compute grid spacing
    [X, Y] = meshgrid(x, y);                     % Generate a 2D meshgrid

    % Evaluate initial conditions on the interior grid points
    u0 = sin(X(2:end,2:end)) .* sin(Y(2:end,2:end));

    % Calculate the number of time steps required to reach t_final
    num_steps = round(t_final/dt);

    % Initialize matrices for the finite difference scheme
    I = eye(Nx);                                 % Identity matrix of size Nx
    e = ones(Nx, 1);                             % Column vector of ones for constructing Laplacian matrices
    T = spdiags([e -4*e e], [-1 0 1], Nx, Nx);   % Tridiagonal matrix representing 1D Laplacian
    S = spdiags([e e], [-1 1], Nx, Nx);          % Off-diagonal matrix for coupling rows in y-direction
    A = (1/(dx*dx)) * (kron(I, T) + kron(S, I)); % Assemble 2D Laplacian using Kronecker products

    % Flatten initial condition matrix into a vector for computation
    Uint = reshape(u0,(Nx)^2,1);       % Reshape initial temperature distribution into a column vector

    % Time-stepping loop using the explicit Euler method
    for k = 1:num_steps
        Un = A * Uint;                 % Compute the next state using the Laplacian matrix
        Uint = Uint + alpha * dt * Un; % Update the state vector
    end  

    % Append the first column and first row to wrap the boundary conditions
    U = reshape(Uint,Nx,Nx);           % Reshape the final state back into a 2D matrix
    U = [U(:,1) U];
    U = [U(1,:);U ];
end