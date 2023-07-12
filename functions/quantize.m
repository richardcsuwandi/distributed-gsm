function result = quantize(y, Delta)

% Set m to ensure y in [m*Delta, (m + 1)*Delta]
m = floor(y/Delta);

if any(y < m*Delta) || any(y > (m + 1)*Delta)
    disp('y is out of range')
    result = y;
else
    % Return m*Delta w.p. 1 - (y/Delta - m) and (m+1)*Delta w.p. y/Delta - m
    if rand() < (y/Delta - m)
        result = (m + 1)*Delta;
    else
        result = m*Delta;
    end
end

end