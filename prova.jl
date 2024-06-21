using NPZ, JLD2, GraphNeuralNetworks, Flux, CUDA, Statistics

data = npzread("/user/aurossi/home/Downloads/mnist_test_seq_toronto.npy")

A = JLD2.load("/user/aurossi/home/Projects/GNNPractice/movingMNIST/A.jld2")["A"]

function myvec(x)
    a = zeros(size(x,1),size(x,2)*size(x,3))
    for i in 1:size(x,1)
        x =permutedims(x,(1,3,2))
        a[i,:] = vec(x[i,:,:])
    end
    a = reshape(a,(1,size(a,2),size(a,1)))
    b = zeros(256,size(x,2)*size(x,3),size(x,1))
    for k in 1:size(a,2)
        for t in 1:size(x,1)
            b[:,k,t] = Flux.onehot(a[1,k,t], 0:255)
        end
    end
    return b
end


function create_dataset(data)
    features = []
    targets = []
    for i in 1:200
        push!(features, myvec(data[1:10,i,:,:]))
        push!(targets, myvec(data[11:20,i,:,:]))
    end
    return features, targets
end




features, targets = create_dataset(Int.(data))




model = GNNChain(GraphNeuralNetworks.GConvGRU(256=> 1,2,4096), Dense(256,256)) 

trainloader = zip(features[1:3],targets[1:3]) 

g = GNNGraph(A)


function train(graph, train_loader, model)

    opt = Flux.setup(Adam(0.001), model)

    for epoch in 1:100
        for (x, y) in train_loader
            x, y = (x, y)
            grads = Flux.gradient(model) do model
                ŷ = model(graph, x)
                Flux.logitcrossentropy(ŷ, y) 
            end
            Flux.update!(opt, model, grads[1])
        end
        
        if epoch % 10 == 0
            loss = mean([Flux.logitcrossentropy(model(graph,x), y) for (x, y) in train_loader])
            @show epoch, loss
        end
    end
    return model
end

train(g, trainloader, model)