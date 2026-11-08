function eta_s_values = calcular_eta_s_newton(t_values, M, chi_s)
% Calcula eta_s usando Newton-Raphson para cada valor de t

% Parámetros de Newton-Raphson
tol = 1e-10;
max_iter = 50;

eta_s_values = NaN(size(t_values));

for i = 1:length(t_values)
    t = t_values(i);
    
    % Semilla inicial: -pi/2 es buen valor
    eta_s = -pi/2;
    
    for iter = 1:max_iter
        % Evaluar F y su derivada
        [F_val, dF_val] = eval_F_and_dF(eta_s, M, chi_s, t);
        
        if ~isfinite(F_val) || ~isfinite(dF_val)
            warning('Valor no finito en iteración.');
            break;
        end
        
        % Newton-Raphson update
        eta_s_new = eta_s - F_val / dF_val;
        
        % Comprobar convergencia
        if abs(eta_s_new - eta_s) < tol
            eta_s = eta_s_new;
            break;
        end
        
        eta_s = eta_s_new;
    end
    
    if iter == max_iter
        warning('No converge en t=%.5f. Intentando fzero...', t);
        
        % Fallback a fzero si falla Newton
        eta_s = calcular_eta_s_fzero(t, M, chi_s);
    end
    
    eta_s_values(i) = eta_s;
end

end