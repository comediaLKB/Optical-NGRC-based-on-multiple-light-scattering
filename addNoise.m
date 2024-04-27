function noisy_speckle = addNoise(I_speckle, noise_std_per_mean)
    noise_std = mean2(I_speckle) * noise_std_per_mean;
    noise_mask = noise_std * randn(size(I_speckle,1), size(I_speckle,2));
    noisy_speckle = I_speckle + noise_mask;
    noisy_speckle(noisy_speckle<0) = 0;
%     noisy_speckle = abs(noisy_speckle);
end