var documenterSearchIndex = {"docs":
[{"location":"api/messagepassing/","page":"Message Passing","title":"Message Passing","text":"CurrentModule = GNNlib","category":"page"},{"location":"api/messagepassing/#Message-Passing","page":"Message Passing","title":"Message Passing","text":"","category":"section"},{"location":"api/messagepassing/#Index","page":"Message Passing","title":"Index","text":"","category":"section"},{"location":"api/messagepassing/","page":"Message Passing","title":"Message Passing","text":"Order = [:type, :function]\nPages   = [\"messagepassing.md\"]","category":"page"},{"location":"api/messagepassing/#Interface","page":"Message Passing","title":"Interface","text":"","category":"section"},{"location":"api/messagepassing/","page":"Message Passing","title":"Message Passing","text":"GNNlib.apply_edges\nGNNlib.aggregate_neighbors\nGNNlib.propagate","category":"page"},{"location":"api/messagepassing/#GNNlib.apply_edges","page":"Message Passing","title":"GNNlib.apply_edges","text":"apply_edges(fmsg, g; [xi, xj, e])\napply_edges(fmsg, g, xi, xj, e=nothing)\n\nReturns the message from node j to node i applying the message function fmsg on the edges in graph g. In the message-passing scheme, the incoming messages  from the neighborhood of i will later be aggregated in order to update the features of node i (see aggregate_neighbors).\n\nThe function fmsg operates on batches of edges, therefore xi, xj, and e are tensors whose last dimension is the batch size, or can be named tuples of  such tensors.\n\nArguments\n\ng: An AbstractGNNGraph.\nxi: An array or a named tuple containing arrays whose last dimension's size        is g.num_nodes. It will be appropriately materialized on the       target node of each edge (see also edge_index).\nxj: As xi, but now to be materialized on each edge's source node. \ne: An array or a named tuple containing arrays whose last dimension's size is g.num_edges.\nfmsg: A function that takes as inputs the edge-materialized xi, xj, and e.      These are arrays (or named tuples of arrays) whose last dimension' size is the size of      a batch of edges. The output of f has to be an array (or a named tuple of arrays)      with the same batch size. If also layer is passed to propagate,     the signature of fmsg has to be fmsg(layer, xi, xj, e)      instead of fmsg(xi, xj, e).\n\nSee also propagate and aggregate_neighbors.\n\n\n\n\n\n","category":"function"},{"location":"api/messagepassing/#GNNlib.aggregate_neighbors","page":"Message Passing","title":"GNNlib.aggregate_neighbors","text":"aggregate_neighbors(g, aggr, m)\n\nGiven a graph g, edge features m, and an aggregation operator aggr (e.g +, min, max, mean), returns the new node features \n\nmathbfx_i = square_j in mathcalN(i) mathbfm_jto i\n\nNeighborhood aggregation is the second step of propagate,  where it comes after apply_edges.\n\n\n\n\n\n","category":"function"},{"location":"api/messagepassing/#GNNlib.propagate","page":"Message Passing","title":"GNNlib.propagate","text":"propagate(fmsg, g, aggr; [xi, xj, e])\npropagate(fmsg, g, aggr xi, xj, e=nothing)\n\nPerforms message passing on graph g. Takes care of materializing the node features on each edge,  applying the message function fmsg, and returning an aggregated message barmathbfm  (depending on the return value of fmsg, an array or a named tuple of  arrays with last dimension's size g.num_nodes).\n\nIt can be decomposed in two steps:\n\nm = apply_edges(fmsg, g, xi, xj, e)\nm̄ = aggregate_neighbors(g, aggr, m)\n\nGNN layers typically call propagate in their forward pass, providing as input f a closure.  \n\nArguments\n\ng: A GNNGraph.\nxi: An array or a named tuple containing arrays whose last dimension's size        is g.num_nodes. It will be appropriately materialized on the       target node of each edge (see also edge_index).\nxj: As xj, but to be materialized on edges' sources. \ne: An array or a named tuple containing arrays whose last dimension's size is g.num_edges.\nfmsg: A generic function that will be passed over to apply_edges.      Has to take as inputs the edge-materialized xi, xj, and e      (arrays or named tuples of arrays whose last dimension' size is the size of      a batch of edges). Its output has to be an array or a named tuple of arrays     with the same batch size. If also layer is passed to propagate,     the signature of fmsg has to be fmsg(layer, xi, xj, e)      instead of fmsg(xi, xj, e).\naggr: Neighborhood aggregation operator. Use +, mean, max, or min. \n\nExamples\n\nusing GraphNeuralNetworks, Flux\n\nstruct GNNConv <: GNNLayer\n    W\n    b\n    σ\nend\n\nFlux.@layer GNNConv\n\nfunction GNNConv(ch::Pair{Int,Int}, σ=identity)\n    in, out = ch\n    W = Flux.glorot_uniform(out, in)\n    b = zeros(Float32, out)\n    GNNConv(W, b, σ)\nend\n\nfunction (l::GNNConv)(g::GNNGraph, x::AbstractMatrix)\n    message(xi, xj, e) = l.W * xj\n    m̄ = propagate(message, g, +, xj=x)\n    return l.σ.(m̄ .+ l.bias)\nend\n\nl = GNNConv(10 => 20)\nl(g, x)\n\nSee also apply_edges and aggregate_neighbors.\n\n\n\n\n\n","category":"function"},{"location":"api/messagepassing/#Built-in-message-functions","page":"Message Passing","title":"Built-in message functions","text":"","category":"section"},{"location":"api/messagepassing/","page":"Message Passing","title":"Message Passing","text":"GNNlib.copy_xi\nGNNlib.copy_xj\nGNNlib.xi_dot_xj\nGNNlib.xi_sub_xj\nGNNlib.xj_sub_xi\nGNNlib.e_mul_xj\nGNNlib.w_mul_xj","category":"page"},{"location":"api/messagepassing/#GNNlib.copy_xi","page":"Message Passing","title":"GNNlib.copy_xi","text":"copy_xi(xi, xj, e) = xi\n\n\n\n\n\n","category":"function"},{"location":"api/messagepassing/#GNNlib.copy_xj","page":"Message Passing","title":"GNNlib.copy_xj","text":"copy_xj(xi, xj, e) = xj\n\n\n\n\n\n","category":"function"},{"location":"api/messagepassing/#GNNlib.xi_dot_xj","page":"Message Passing","title":"GNNlib.xi_dot_xj","text":"xi_dot_xj(xi, xj, e) = sum(xi .* xj, dims=1)\n\n\n\n\n\n","category":"function"},{"location":"api/messagepassing/#GNNlib.xi_sub_xj","page":"Message Passing","title":"GNNlib.xi_sub_xj","text":"xi_sub_xj(xi, xj, e) = xi .- xj\n\n\n\n\n\n","category":"function"},{"location":"api/messagepassing/#GNNlib.xj_sub_xi","page":"Message Passing","title":"GNNlib.xj_sub_xi","text":"xj_sub_xi(xi, xj, e) = xj .- xi\n\n\n\n\n\n","category":"function"},{"location":"api/messagepassing/#GNNlib.e_mul_xj","page":"Message Passing","title":"GNNlib.e_mul_xj","text":"e_mul_xj(xi, xj, e) = reshape(e, (...)) .* xj\n\nReshape e into broadcast compatible shape with xj (by prepending singleton dimensions) then perform broadcasted multiplication.\n\n\n\n\n\n","category":"function"},{"location":"api/messagepassing/#GNNlib.w_mul_xj","page":"Message Passing","title":"GNNlib.w_mul_xj","text":"w_mul_xj(xi, xj, w) = reshape(w, (...)) .* xj\n\nSimilar to e_mul_xj but specialized on scalar edge features (weights).\n\n\n\n\n\n","category":"function"},{"location":"api/utils/","page":"Utils","title":"Utils","text":"CurrentModule = GNNlib","category":"page"},{"location":"api/utils/#Utility-Functions","page":"Utils","title":"Utility Functions","text":"","category":"section"},{"location":"api/utils/#Index","page":"Utils","title":"Index","text":"","category":"section"},{"location":"api/utils/","page":"Utils","title":"Utils","text":"Order = [:type, :function]\nPages   = [\"utils.md\"]","category":"page"},{"location":"api/utils/#Docs","page":"Utils","title":"Docs","text":"","category":"section"},{"location":"api/utils/#Graph-wise-operations","page":"Utils","title":"Graph-wise operations","text":"","category":"section"},{"location":"api/utils/","page":"Utils","title":"Utils","text":"reduce_nodes\nreduce_edges\nsoftmax_nodes\nsoftmax_edges\nbroadcast_nodes\nbroadcast_edges","category":"page"},{"location":"api/utils/#GNNlib.reduce_nodes","page":"Utils","title":"GNNlib.reduce_nodes","text":"reduce_nodes(aggr, g, x)\n\nFor a batched graph g, return the graph-wise aggregation of the node features x. The aggregation operator aggr can be +, mean, max, or min. The returned array will have last dimension g.num_graphs.\n\nSee also: reduce_edges.\n\n\n\n\n\nreduce_nodes(aggr, indicator::AbstractVector, x)\n\nReturn the graph-wise aggregation of the node features x given the graph indicator indicator. The aggregation operator aggr can be +, mean, max, or min.\n\nSee also graph_indicator.\n\n\n\n\n\n","category":"function"},{"location":"api/utils/#GNNlib.reduce_edges","page":"Utils","title":"GNNlib.reduce_edges","text":"reduce_edges(aggr, g, e)\n\nFor a batched graph g, return the graph-wise aggregation of the edge features e. The aggregation operator aggr can be +, mean, max, or min. The returned array will have last dimension g.num_graphs.\n\n\n\n\n\n","category":"function"},{"location":"api/utils/#GNNlib.softmax_nodes","page":"Utils","title":"GNNlib.softmax_nodes","text":"softmax_nodes(g, x)\n\nGraph-wise softmax of the node features x.\n\n\n\n\n\n","category":"function"},{"location":"api/utils/#GNNlib.softmax_edges","page":"Utils","title":"GNNlib.softmax_edges","text":"softmax_edges(g, e)\n\nGraph-wise softmax of the edge features e.\n\n\n\n\n\n","category":"function"},{"location":"api/utils/#GNNlib.broadcast_nodes","page":"Utils","title":"GNNlib.broadcast_nodes","text":"broadcast_nodes(g, x)\n\nGraph-wise broadcast array x of size (*, g.num_graphs)  to size (*, g.num_nodes).\n\n\n\n\n\n","category":"function"},{"location":"api/utils/#GNNlib.broadcast_edges","page":"Utils","title":"GNNlib.broadcast_edges","text":"broadcast_edges(g, x)\n\nGraph-wise broadcast array x of size (*, g.num_graphs)  to size (*, g.num_edges).\n\n\n\n\n\n","category":"function"},{"location":"api/utils/#Neighborhood-operations","page":"Utils","title":"Neighborhood operations","text":"","category":"section"},{"location":"api/utils/","page":"Utils","title":"Utils","text":"softmax_edge_neighbors","category":"page"},{"location":"api/utils/#GNNlib.softmax_edge_neighbors","page":"Utils","title":"GNNlib.softmax_edge_neighbors","text":"softmax_edge_neighbors(g, e)\n\nSoftmax over each node's neighborhood of the edge features e.\n\nmathbfe_jto i = frace^mathbfe_jto i\n                    sum_jin N(i) e^mathbfe_jto i\n\n\n\n\n\n","category":"function"},{"location":"api/utils/#NNlib","page":"Utils","title":"NNlib","text":"","category":"section"},{"location":"api/utils/","page":"Utils","title":"Utils","text":"Primitive functions implemented in NNlib.jl:","category":"page"},{"location":"api/utils/","page":"Utils","title":"Utils","text":"gather!\ngather\nscatter!\nscatter","category":"page"},{"location":"messagepassing/#Message-Passing","page":"Message Passing","title":"Message Passing","text":"","category":"section"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"A generic message passing on graph takes the form","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"beginaligned\nmathbfm_jto i = phi(mathbfx_i mathbfx_j mathbfe_jto i) \nbarmathbfm_i = square_jin N(i)  mathbfm_jto i \nmathbfx_i = gamma_x(mathbfx_i barmathbfm_i)\nmathbfe_jto i^prime =  gamma_e(mathbfe_j to imathbfm_j to i)\nendaligned","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"where we refer to phi as to the message function,  and to gamma_x and gamma_e as to the node update and edge update function respectively. The aggregation square is over the neighborhood N(i) of node i,  and it is usually equal either to sum, to max or to a mean operation. ","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"In GNNlib.jl, the message passing mechanism is exposed by the propagate function. propagate takes care of materializing the node features on each edge, applying the message function, performing the aggregation, and returning barmathbfm.  It is then left to the user to perform further node and edge updates, manipulating arrays of size D_node times num_nodes and    D_edge times num_edges.","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"propagate is composed of two steps, also available as two independent methods:","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"apply_edges materializes node features on edges and applies the message function. \naggregate_neighbors applies a reduction operator on the messages coming from the neighborhood of each node.","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"The whole propagation mechanism internally relies on the NNlib.gather  and NNlib.scatter methods.","category":"page"},{"location":"messagepassing/#Examples","page":"Message Passing","title":"Examples","text":"","category":"section"},{"location":"messagepassing/#Basic-use-of-apply_edges-and-propagate","page":"Message Passing","title":"Basic use of apply_edges and propagate","text":"","category":"section"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"The function apply_edges can be used to broadcast node data on each edge and produce new edge data.","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"julia> using GNNlib, Graphs, Statistics\n\njulia> g = rand_graph(10, 20)\nGNNGraph:\n    num_nodes = 10\n    num_edges = 20\n\njulia> x = ones(2,10);\n\njulia> z = 2ones(2,10);\n\n# Return an edge features arrays (D × num_edges)\njulia> apply_edges((xi, xj, e) -> xi .+ xj, g, xi=x, xj=z)\n2×20 Matrix{Float64}:\n 3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0\n 3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0\n\n# now returning a named tuple\njulia> apply_edges((xi, xj, e) -> (a=xi .+ xj, b=xi .- xj), g, xi=x, xj=z)\n(a = [3.0 3.0 … 3.0 3.0; 3.0 3.0 … 3.0 3.0], b = [-1.0 -1.0 … -1.0 -1.0; -1.0 -1.0 … -1.0 -1.0])\n\n# Here we provide a named tuple input\njulia> apply_edges((xi, xj, e) -> xi.a + xi.b .* xj, g, xi=(a=x,b=z), xj=z)\n2×20 Matrix{Float64}:\n 5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0\n 5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"The function propagate instead performs the apply_edges operation but then also applies a reduction over each node's neighborhood (see aggregate_neighbors).","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"julia> propagate((xi, xj, e) -> xi .+ xj, g, +, xi=x, xj=z)\n2×10 Matrix{Float64}:\n 3.0  6.0  9.0  9.0  0.0  6.0  6.0  3.0  15.0  3.0\n 3.0  6.0  9.0  9.0  0.0  6.0  6.0  3.0  15.0  3.0\n\n# Previous output can be understood by looking at the degree\njulia> degree(g)\n10-element Vector{Int64}:\n 1\n 2\n 3\n 3\n 0\n 2\n 2\n 1\n 5\n 1","category":"page"},{"location":"messagepassing/#Implementing-a-custom-Graph-Convolutional-Layer-using-Flux.jl","page":"Message Passing","title":"Implementing a custom Graph Convolutional Layer using Flux.jl","text":"","category":"section"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"Let's implement a simple graph convolutional layer using the message passing framework using the machine learning framework Flux.jl. The convolution reads ","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"mathbfx_i = W cdot sum_j in N(i)  mathbfx_j","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"We will also add a bias and an activation function.","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"using Flux, Graphs, GraphNeuralNetworks\n\nstruct GCN{A<:AbstractMatrix, B, F} <: GNNLayer\n    weight::A\n    bias::B\n    σ::F\nend\n\nFlux.@layer GCN # allow gpu movement, select trainable params etc...\n\nfunction GCN(ch::Pair{Int,Int}, σ=identity)\n    in, out = ch\n    W = Flux.glorot_uniform(out, in)\n    b = zeros(Float32, out)\n    GCN(W, b, σ)\nend\n\nfunction (l::GCN)(g::GNNGraph, x::AbstractMatrix{T}) where T\n    @assert size(x, 2) == g.num_nodes\n\n    # Computes messages from source/neighbour nodes (j) to target/root nodes (i).\n    # The message function will have to handle matrices of size (*, num_edges).\n    # In this simple case we just let the neighbor features go through.\n    message(xi, xj, e) = xj \n\n    # The + operator gives the sum aggregation.\n    # `mean`, `max`, `min`, and `*` are other possibilities.\n    x = propagate(message, g, +, xj=x) \n\n    return l.σ.(l.weight * x .+ l.bias)\nend","category":"page"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"See the GATConv implementation here for a more complex example.","category":"page"},{"location":"messagepassing/#Built-in-message-functions","page":"Message Passing","title":"Built-in message functions","text":"","category":"section"},{"location":"messagepassing/","page":"Message Passing","title":"Message Passing","text":"In order to exploit optimized specializations of the propagate, it is recommended  to use built-in message functions such as copy_xj whenever possible. ","category":"page"},{"location":"#GNNlib.jl","page":"Home","title":"GNNlib.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"GNNlib.jl is a package that provides the implementation of the basic message passing functions and  functional implementation of graph convolutional layers, which are used to build graph neural networks in both the Flux.jl and Lux.jl machine learning frameworks, created in the GraphNeuralNetworks.jl and GNNLux.jl packages, respectively.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package depends on GNNGraphs.jl and NNlib.jl, and is primarily intended for developers looking to create new GNN architectures. For most users, the higher-level GraphNeuralNetworks.jl and GNNLux.jl packages are recommended.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add GNNlib","category":"page"}]
}
