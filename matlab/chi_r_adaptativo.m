function [chi_matrix, r_matrix] = chi_r_adaptativo(eta_c_values, R0, chi_s, n_pts_r)
% Calcula chi(r,t) con r adaptativo en cada instante
%
% Entradas:
%   t_values: vector de tiempos
%   eta_c_values: vector de eta_c en los instantes de tiempo de t_values
%   R0: radio inicial
%   chi_s: coordenada radial comóvil en la superficie
%   n_pts_r: número de puntos de r por tiempo
%
% Salidas:
%   chi_matrix: matriz (long(t_values) x 1500) de chi
%   r_matrix: matriz (long(t_values) x 1500) de radios correspondientes

n_times = length(eta_c_values);

% Prealocar matrices
chi_matrix = NaN(n_times, n_pts_r);
r_matrix = NaN(n_times, n_pts_r);

% Paralelizar sobre tiempos
parfor it = 1:n_times % poner parfor
    eta_c = eta_c_values(it);
    
    % 1. Radio de la estrella en este instante
    R_t = R0 * (1 - cos(eta_c/2)^2 / cos(chi_s));
    
    % 2. Crear vector adaptativo de radios
    r_values = logspace(-3, log10(R_t), n_pts_r); % OJO: no poner un r_min
    % demasiado pequeño porque en ese caso el train de la PINN explota (en
    % las EDP aparece r^2 dividiendo)
    r_matrix(it,:) = r_values; % Guardar
    
    % 3. Resolver chi para cada r
    chi_row = calcular_chi_fzero(r_values, eta_c, R0, chi_s);
    
    % 4. Guardar la fila
    chi_matrix(it,:) = chi_row;
end

end