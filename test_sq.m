addpath('./functions/')

n = 100; % Number of data

% Generate random data
y = randn(n, 1);

Delta_vals = [0.2, 0.5, 1, 2];
for i = 1:length(Delta_vals)
    % Quantize the data
    Delta = Delta_vals(i); % Quantization resolution
    y_Q = quantize(y, Delta);
    error = abs(y - y_Q);

    % Calculate the number of bits for original y and quantized y
    bits_y = numel(y) * 64; % Assuming double precision (64 bits per element)
    bits_y_Q = numel(y_Q) * log2((max(y) - min(y)) / Delta + 1); % Bits needed to represent the quantized values

    % Plot the data vs. quantized data
    figure(); hold on;
    plot(1:n, y, 'Color', 'blue');
    plot(1:n, y_Q, 'Color', 'red');
    plot(1:n, error, 'Color', 'green')
    title("Delta = " + Delta)

    % Display the number of bits
    disp("Number of bits for y: " + bits_y)
    disp("Number of bits for y_Q: " + bits_y_Q)
end