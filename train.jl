# Training ASPIRE

# TODO: Draw N samples: x

# TODO: Simulate x -> y

# TODO: Init N Fiducials: x0

for j = 1:J
    # TODO: Calculate summary statistic yhat for all observations y around fiducials x0

    # TODO: Train CNF on (x, yhat)

    # TODO: Update fiducials x0 by averaging posterior samples conditioned on yhats

end

# Inference ASPIRE

# TODO: Init yobs and x0

for j = 1:J
    # TODO: Calculate summary statistic yhat for yobs around fiducials x0

    # TODO: Update fiducials x0 by averaging posterior samples conditioned on yhat
    
end
