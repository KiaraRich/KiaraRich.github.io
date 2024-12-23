% implementing the finite difference method on 2D heat equation
clc; clear;
% Initaliazing some variables
L = 2*pi;
t_final = 1;
Nvec = [20,40,60,80,100];
m = length(Nvec);
errf = zeros(m,1);

for j = 1:m
 h = L/(Nvec(j)+1);  
Nx = Nvec(j);
Ny = Nvec(j);

alpha = 1;
%u0 = @(x,y) sin(x).*sin(y); % initial condition
dt = 0.00075; % Time step size

% solving with finite diference method
x = linspace(0,L,Nx+1);
y = linspace(0,L,Ny+1);
[X,Y] = meshgrid(x,y);
U = finite_diff(L,Nx,Ny,alpha, dt, t_final);

% exact soluiton
u_true = @(x,y, t) sin(x).*sin(y)*exp(-2*t);
u_tval = u_true(X,Y,t_final);

%  Calculating the error in the spectral methods

errf(j) = sqrt(h*h*sum((U-u_tval).^2, 'all'));

end

% plot the solutions
figure (1);
surf(X,Y,U); 
xlabel('X');
ylabel('Y');
zlabel('Temperature');
title('Heat Equation 2D: Finite Difference Method')

figure (2);
surf(X,Y,u_tval); 
xlabel('X');
ylabel('Y');
zlabel('Temperature');
title('Heat Equation 2D: True solution')

figure(3);
plot(1:m,errf,'r')
xlabel('x');
ylabel('Error');
title('Convergence History')



function U = finite_diff(L,Nx,Ny,alpha, dt, t_final)
    % Inputs:
    % - L: Length of the domain
    % - Nx: Number of grid points along the x-axis
    % - Ny: Number of grid points along the y-axis
    % - alpha: Thermal diffusivity constant
    % - t_final: Final time for simulation
    %
    % Output:
    % - U: Solution of the heat equation at the final time t_final, represented as a 2D grid

    % Spatial grid to discretize
    x = linspace(0,L,Nx+1); % Create spatial grid along x-axis
    y = linspace(0,L,Ny+1); % Create spatial grid along y-axis
    dx = L/(Nx+1); % Step size of meshgrid (width) along x-axis
    [X,Y] = meshgrid(x,y); % Create meshgrid for 2D domain

    % Initial conditions
    u0 = sin(X(2:end,2:end)) .* sin(Y(2:end,2:end));

    % Number of time steps
    num_steps = round(t_final/dt); % Calculate number of time steps based on final time and time step size

    % Initializing the matrix
    I = eye(Nx); % Identity matrix of size Nx
    e = ones(Nx,1); % Vector of ones of size Nx
    T = spdiags([e -4*e e],[-1 0 1],Nx,Nx); % Tridiagonal matrix for 1D Laplacian
    S = spdiags([e e],[-1 1],Nx,Nx); % Secondary tridiagonal matrix for 1D Laplacian
    A = (1/(dx*dx))*(kron(I,T) + kron(S,I)); % 2D Laplacian matrix constructed by Kronecker product

    Uint = reshape(u0,(Nx)^2,1); % Reshape initial temperature distribution into a column vector

    % Time stepping loop
    for k = 1:num_steps
        Un = A * Uint; % Calculate temperature distribution at next time step
        Uint = Uint + alpha * dt * Un; % Update temperature distribution using forward Euler method
    end     
    U = reshape(Uint,Nx,Nx); % Reshape temperature distribution into a 2D grid
    U = [U(:,1) U];
    U = [U(1,:);U ];
end