function [bits_original, bits_quantized, savings] = compute_bits(x, Delta)

d = length(x);
xmax = max(x); xmin = min(x);

bits_original = 64 * d;
bits_quantized = d * log2((xmax - xmin)/Delta + 1);

savings = bits_original / bits_quantized;

end