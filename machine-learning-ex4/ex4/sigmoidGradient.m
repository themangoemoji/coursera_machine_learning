function g = sigmoidGradient(z)

    function res = comp_g(input_z)
        denom = 1 + (e .^ -input_z);
        res = 1 ./ denom;
    end


    g = comp_g(z) .* (1 - comp_g(z));


% =============================================================

end
