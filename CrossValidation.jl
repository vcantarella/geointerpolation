using Random
include("Variogram.jl")
include("VariogramModels.jl")
include("Kriging.jl")

function cross_validate_kriging(data, variable, n_folds=5,
    model::Function,
    p::Vector{T},)
    # Split data into n_folds
    folds = randperm(nrow(data)) |> x -> reshape(x, :, n_folds)
    
    errors = Float64[]
    
    for i in 1:n_folds
        # Separate training and validation sets
        val_idx = folds[:, i]
        train_idx = setdiff(1:nrow(data), val_idx)
        
        train_data = data[train_idx, :]
        val_data = data[val_idx, :]
        
        # Fit kriging model
        model = Kriging(variable ~ 1, train_data)
        
        # Predict on validation set
        predictions = predict(model, val_data)
        
        # Calculate error (e.g., Mean Squared Error)
        mse = mean((predictions - val_data[variable]).^2)
        push!(errors, mse)
    end
    
    # Return average error
    return mean(errors)
end

# Example usage
# data = ... # Load your data here
# variable = :your_variable_name
# avg_error = cross_validate_kriging(data, variable)
# println("Average cross-validation error: $avg_error")