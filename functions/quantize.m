function x_Q = quantize(x, Delta)

d = length(x);
x_Q = zeros(d, 1);

for i = 1:d
    xi = x(i);
    m = floor(xi/Delta);
    r = xi/Delta - m;
    
    if rand() < r
        x_Q(i) = (m + 1) * Delta;
    else
        x_Q(i) = m * Delta;
    end
end

end