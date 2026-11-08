function t_calc = ecuacion46(eta_s, M, chi_s)
% Calcula el valor de t segun la ecuacion (46) de Romero

term1 = (1 / tan(chi_s)) * (eta_s + pi + (0.5) * sin(chi_s)^(-2) * (eta_s + pi - sin(eta_s)));
term2 = log((tan(eta_s/2) - tan(chi_s)) / (tan(eta_s/2) + tan(chi_s)));

t_calc = 2 * M * (term1 + term2);

end