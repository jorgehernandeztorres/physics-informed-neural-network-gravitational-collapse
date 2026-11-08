function eta_s_values = calcular_eta_s_fzero(t_values, M, chi_s)
% Esta funcion calcula eta_s para cada valor de t dado
%
% Entrada:
%   t_values: vector de tiempos t (puede ser escalar)
%   M: masa total (en unidades coherentes)
%   chi_s: coordenada radial comóvil en la superficie
%
% Salida:
%   eta_s_values: vector de valores eta_s correspondientes a t_values

% Preasigna vector de resultados
eta_s_values = zeros(size(t_values));

% --- Búsqueda robusta ---
% Buscar eta_s en un intervalo seguro [-pi, 0]
eta_s_guess_range = linspace(-pi - 1e-4, +1e-4, 1000);

% Opciones para fzero
options = optimset('Display', 'off');

parfor i = 1:length(t_values)
    t = t_values(i);

    % Función objetivo: F(eta_s) = t(eta_s) - t
    F = @(eta_s) ecuacion46(eta_s, M, chi_s) - t;

    % Evaluar F en ese rango
    F_vals = arrayfun(F, eta_s_guess_range);

    % Filtrar solo reales (permitiendo pequeñas imprecisiones numéricas)
    idx_reales = abs(imag(F_vals)) < 1e-12;

    % Seleccionar valores finitos y reales
    valid_idx = find(isfinite(F_vals(idx_reales)));
    F_vals_valid = F_vals(idx_reales(valid_idx));
    eta_s_guess_range_valid = eta_s_guess_range(idx_reales(valid_idx));

    if isempty(valid_idx)
        error('No hay valores iniciales válidos para fzero.');
    end

    % Encuentra cruces de signo
    idx_cross = find(diff(sign(F_vals_valid)) ~= 0);
    
    if isempty(idx_cross)
        error('No se ha encontrado ningún cruce de signo.');
    end
    
    % Calcula diferencia en F entre extremos del cruce
    delta_F = abs(F_vals_valid(idx_cross + 1) - F_vals_valid(idx_cross));
    
    % Elige el cruce donde delta_F sea mínimo
    [~, best_idx] = min(delta_F);
    
    % Intervalo final para fzero
    eta_s_ini = eta_s_guess_range_valid(idx_cross(best_idx));
    eta_s_fin = eta_s_guess_range_valid(idx_cross(best_idx) + 1);
    
    % Resolver usando fzero
    eta_s_values(i) = fzero(F, [eta_s_ini, eta_s_fin], options);
end

end