M = 1;
R0 = 10;

if 2*M > R0
    error('El radio inicial ha de ser mayor que el radio de Schwarzschild.')
end

chi_s = asin(sqrt(2*M/R0));


%% CÁLCULO DE t_max ( es aquel para el que R(t_max)=2*M=R_{Schwarzcshild} )

eta_c_max = - 2 * acos(sqrt(cos(chi_s) * (1 - sin(chi_s)^2)));
eta_s_max = relacion_eta_s(eta_c_max, chi_s);
t_max = real(ecuacion46(eta_s_max, M, chi_s));
% esta condición es equivalente a considerar eta_s_max = -2*chi_s, pues eso
% lleva en la ecuación (46) a tener un ln(0) y por tanto una divergencia de
% t

if isinf(t_max)
    % Si esto se da, y según Petrich ocurrirá siempre, entonces tomamos
    t_max = 50; % arbitrario; para M=1 y R0=10 da R_end=2.0256≈2M

    % Para M = 1 masa solar, de acuerdo con Gourg, t_max = 224.9e-6 se
    % obtiene un R(t_max) = 2.21M, cercano al radio de Shcwarzschild
end


% Número de puntos de r por tiempo
n_pts_r = 500; % poner 1500

% Vector de tiempos
n_times = 51; % poner 201
t_vector = linspace(0,t_max,n_times);


%% CÁLCULO DE chi Y r

% 1. Invertir t=t(eta_s)
eta_s_values = calcular_eta_s_fzero(t_vector, M, chi_s);
% Se utiliza fzero en lugar de Newton-Raphson para evitar
% depender de una derivada que no ha sido verificada

% 2. Hallar eta_c=eta_c(eta_s)
eta_c_values = relacion_eta_c(eta_s_values, chi_s);

% 3. Radio de la estrella en el instante final
R_end = R0 * (1 - cos(eta_c_values(end)/2)^2 / cos(chi_s));

% 4. Obtener de aquí las matrices de chi y de r
%    { se cumple dim(r) = length(t_vector) x n_pts_r, y
%      r(i,end) = R(t_vector(i)) para todo 1≤i≤length(t_vector) }
[chi_matrix, r_matrix] = chi_r_adaptativo(eta_c_values, R0, chi_s, n_pts_r);


%% CÁLCULO DE eta (ecuación 3.21 de Petrich) ---> realmente no hace falta

% eta_matrix = NaN(n_times, n_pts_r);
% 
% for i=1:n_times
%     eta_c = eta_c_values(i);
%     eta_matrix(i,:) = - 2 * acos(cos(eta_c/2) ./ sqrt(cos(chi_matrix(i,:))));
% end


%% CÁLCULO DE alpha

alpha_matrix = NaN(n_times, n_pts_r);

for i=1:n_times
    eta_c = eta_c_values(i);
    chi = chi_matrix(i,:);
    alpha_0 = (cos(chi_s)^3 - cos(eta_c/2)^2) / (cos(chi_s) - cos(eta_c/2)^2)^(3/2);
    alpha_matrix(i,:) = alpha_0 * (cos(chi) - cos(eta_c/2)^2) ./ sqrt(cos(chi).^3 - cos(eta_c/2)^2);
end


%% CÁLCULO DE X

X_matrix = NaN(n_times, n_pts_r);

for i=1:n_times
    eta_c = eta_c_values(i);
    chi = chi_matrix(i,:);
    X_matrix(i,:) = sqrt( (cos(chi) - cos(eta_c/2)^2) ./ (cos(chi).^3 - cos(eta_c/2)^2) );
end


%% CÁLCULO DE v

v_matrix = NaN(n_times, n_pts_r);

for i=1:n_times
    eta_c = eta_c_values(i);
    chi = chi_matrix(i,:);
    v_matrix(i,:) = - (cos(eta_c/2) * tan(chi)) ./ sqrt(cos(chi) - cos(eta_c/2)^2);
end


%% CÁLCULO DE rho0

rho0_matrix = NaN(n_times, n_pts_r);

for i=1:n_times
    eta_c = eta_c_values(i);
    chi = chi_matrix(i,:);
    rho0_matrix(i,:) = (3 * sin(chi_s)^6) / (32 * pi * M^2) * ( cos(chi) ./ (cos(chi) - cos(eta_c/2)^2) ).^3;
end


%% CÁLCULO DE m

m_matrix = r_matrix/2 .* (1 - 1./X_matrix.^2);


%% CÁLCULO DE Phi

Phi_matrix = log(alpha_matrix);


%% CÁLCULO DE W

W_matrix = 1 ./ sqrt(1 - v_matrix.^2);


%% CÁLCULO DE D

D_matrix = X_matrix .* rho0_matrix .* W_matrix;


%% CÁLCULO DE S ( asumiendo epsilon=0 => h=1 por estar trabajando con polvo )

S_matrix = rho0_matrix .* W_matrix.^2 .* v_matrix;


%% CÁLCULO DE tau

tau_matrix = rho0_matrix .* W_matrix.^2 - D_matrix;


%% CONSTRUCCIÓN FINAL DE u0

alpha_0 = alpha_matrix(1,:)';
X_0 = X_matrix(1,:)';
rho0_0 = rho0_matrix(1,:)';
v_0 = v_matrix(1,:)';

D_0 = D_matrix(1,:)';
S_0 = S_matrix(1,:)';
tau_0 = tau_matrix(1,:)';
m_0 = m_matrix(1,:)';
Phi_0 = Phi_matrix(1,:)';

u0_matrix = [alpha_0 X_0 rho0_0 v_0];

u1_matrix = [D_0 S_0 tau_0 m_0 Phi_0 v_0];

save('OS.mat', 't_vector', 'r_matrix', 'u0_matrix', 'alpha_matrix', ...
    'X_matrix', 'v_matrix', 'rho0_matrix', 'm_matrix', 'Phi_matrix', ...
    'W_matrix', 'D_matrix', 'S_matrix', 'tau_matrix', 'u1_matrix');
