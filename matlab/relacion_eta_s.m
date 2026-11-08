function eta_s = relacion_eta_s(eta_c, chi_s)
% Calcula eta_s a partir de eta_c usando la relacion debajo de la ec. (46)
eta_s = - 2 * acos( cos(eta_c/2) / sqrt(cos(chi_s)) );
end