function r_calc = ecuacion48(chi, eta_c, R0, chi_s)
% Calcula el valor de r segun la ecuacion (48) de Romero

r_calc = R0 * (1 - cos(eta_c/2)^2 / cos(chi)) * sin(chi) / sin(chi_s);

end