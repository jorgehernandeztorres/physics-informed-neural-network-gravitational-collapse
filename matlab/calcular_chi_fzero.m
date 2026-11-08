function chi_values = calcular_chi_fzero(r_values, eta_c, R0, chi_s)
% Esta funcion calcula chi para cada valor de r dado en el instante t
% Entrada:
%   r_values: vector de posiciones r (puede ser escalar)
%   eta_c: valor de eta en el centro de la estrella en el instante t
%   R0: radio inicial de la estrella (en unidades coherentes)
%   chi_s: coordenada radial comóvil en la superficie
% Salida:
%   chi_values: vector de valores chi correspondientes a r_values en el
%   instante t

% Preasigna vector de resultados
chi_values = zeros(size(r_values));

% --- Búsqueda robusta ---
% Buscar chi en un intervalo seguro [0, chi_s]
chi_guess_range = linspace(0, chi_s + 1e-4, 1000);

% Opciones para fzero
options = optimset('Display', 'off');

parfor i = 1:length(r_values)
    r = r_values(i);

    % Función objetivo: F(chi) = r(chi) - r
    F = @(chi) ecuacion48(chi, eta_c, R0, chi_s) - r;

    % Evaluar F en ese rango
    F_vals = arrayfun(F, chi_guess_range);

    % Filtrar solo reales (permitiendo pequeñas imprecisiones numéricas)
    idx_reales = abs(imag(F_vals)) < 1e-12;

    % Seleccionar valores finitos y reales
    valid_idx = find(isfinite(F_vals(idx_reales)));
    F_vals_valid = F_vals(idx_reales(valid_idx));
    chi_guess_range_valid = chi_guess_range(idx_reales(valid_idx));

    if isempty(valid_idx)
        i
        F_vals
        F_vals_valid
        error('No hay valores iniciales válidos para fzero.');
    end

    % Encuentra cruces de signo
    idx_cross = find(diff(sign(F_vals_valid)) ~= 0);
    
    if isempty(idx_cross)
        warning('No se ha encontrado ningún cruce de signo.');
        i
        F_vals
        F_vals_valid
        chi_values(i) = fzero(F, chi_s, options);
        continue;
    end
    
    % Calcula diferencia en F entre extremos del cruce
    delta_F = abs(F_vals_valid(idx_cross + 1) - F_vals_valid(idx_cross));
    
    % Elige el cruce donde delta_F sea mínimo
    [~, best_idx] = min(delta_F);
    
    % Intervalo final para fzero
    chi_ini = chi_guess_range_valid(idx_cross(best_idx));
    chi_fin = chi_guess_range_valid(idx_cross(best_idx) + 1);

    % Resolver usando fzero
    chi_values(i) = fzero(F, [chi_ini, chi_fin], options);
end

end