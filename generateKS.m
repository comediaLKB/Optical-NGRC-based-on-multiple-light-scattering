% cite "Model-Free Prediction of Large Spatiotemporally Chaotic Systems
% from Data: A Reservoir Computing Approach"
clear;clc;
%%
L = 22;  % domain size/periodicity length (denoted by L in Pathak et al)
N = 64; % discretization grid size (denoted by Q)
x = L*(-N/2+1:N/2)'/N;
tmax = 6000; % 4000
h = 1/4; % delta_T
nmax = round(tmax/h);
%%
rng(10)
delta = 0;  % \mu in Eq. (7) REF 1. (set delta = 0 for the standard spatially homogeneous KS equation)
wavelength = L/4;  % \lambda in Eq. (7) REF 1. (sets the spatial inhomgeneity wavelength)
omega = 2*pi/wavelength;
p = delta.*cos(omega.*x);
px = -omega*delta.*sin(omega.*x);
pxx = -(omega^2).*p;
u = 0.6*(-1+2*rand(size(x)));
v = fft(u);
Y = orth(rand(N,num_lyaps));

k = [0:N/2-1 0 -N/2+1:-1]'*(2*pi/L);
L = k.^2 - k.^4;
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);

Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
g = -0.5i*k;

vv = zeros(N, nmax+1); 
vv(:,1) = v;
transient = 1000;
for n = 1:transient
    t = n*h;
    rifftv = real(ifft(v));
    Nv = g.*fft(rifftv.^2) + 2i*k.*fft(rifftv.*px) - fft(rifftv.*pxx) + k.^2.*fft(rifftv.*p);
    a = E2.*v + Q.*Nv;
    riffta = real(ifft(a));
    Na = g.*fft(riffta.^2) + 2i*k.*fft(riffta.*px) - fft(riffta.*pxx) + k.^2.*fft(riffta.*p);
    b = E2.*v + Q.*Na;
    rifftb = real(ifft(b));
    Nb = g.*fft(rifftb.^2) + 2i*k.*fft(rifftb.*px) - fft(rifftb.*pxx) + k.^2.*fft(rifftb.*p);
    c = E2.*a + Q.*(2*Nb-Nv);
    rifftc = real(ifft(c));
    Nc =  g.*fft(rifftc.^2) + 2i.*k.*fft(rifftc.*px) - fft(rifftc.*pxx) + k.^2.*fft(rifftc.*p);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
end

for n=1:nmax+1
    %Evolve KS
    t = n*h;
    rifftv = real(ifft(v));
    Nv = g.*fft(rifftv.^2) + 2i*k.*fft(rifftv.*px) - fft(rifftv.*pxx) + k.^2.*fft(rifftv.*p);
    a = E2.*v + Q.*Nv;
    riffta = real(ifft(a));
    Na = g.*fft(riffta.^2) + 2i*k.*fft(riffta.*px) - fft(riffta.*pxx) + k.^2.*fft(riffta.*p);
    b = E2.*v + Q.*Na;
    rifftb = real(ifft(b));
    Nb = g.*fft(rifftb.^2) + 2i*k.*fft(rifftb.*px) - fft(rifftb.*pxx) + k.^2.*fft(rifftb.*p);
    c = E2.*a + Q.*(2*Nb-Nv);
    rifftc = real(ifft(c));
    Nc =  g.*fft(rifftc.^2) + 2i.*k.*fft(rifftc.*px) - fft(rifftc.*pxx) + k.^2.*fft(rifftc.*p);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
    vv(:,n) = v;
end

uu = transpose(real(ifft(vv)));
% save(['L' num2str(L) '_Ninput' num2str(N) '.mat'], 'uu', 'd', '-v7.3');

figure,
imagesc(transpose(uu(1:2000,:)))
shading flat
colormap(jet);
colorbar;
