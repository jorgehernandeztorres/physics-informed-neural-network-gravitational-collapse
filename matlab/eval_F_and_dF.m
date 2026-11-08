function [F_val, dF_val] = eval_F_and_dF(eta_s, M, chi_s, t)
% Evalúa F y su derivada dF/deta_s

tan_eta2 = tan(eta_s/2);
sec_eta2_sq = 1 + tan_eta2^2;

u = (tan_eta2 - tan(chi_s)) / (tan_eta2 + tan(chi_s));

if (tan_eta2 + tan(chi_s)) == 0 || (tan_eta2 - tan(chi_s)) == 0
    F_val = NaN;
    dF_val = NaN;
    return
end

% F(eta_s)
term1 = (1 / tan(chi_s)) * (eta_s + pi + 0.5 * sin(chi_s)^2 * (eta_s + pi - sin(eta_s)));
term2 = log(u);
F_val = 2*M*(term1 + term2) - t;

% dF/deta_s
dA = (1 / tan(chi_s)) * (1 + 0.5 * sin(chi_s)^2 * (1 - cos(eta_s)));
dB = (tan(chi_s) * sec_eta2_sq) / ( (tan_eta2 - tan(chi_s)) * (tan_eta2 + tan(chi_s)) );

dF_val = 2*M*(dA + dB);
end