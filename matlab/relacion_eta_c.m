function eta_c = relacion_eta_c(eta_s, chi_s)
% Calcula eta_c a partir de eta_s usando la relacion debajo de la ec. (46)
eta_c = - 2 * acos( cos(eta_s/2) * sqrt(cos(chi_s)) );
end