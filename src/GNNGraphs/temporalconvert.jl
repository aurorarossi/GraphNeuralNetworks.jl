function to_coo(coo::COOT_T; dir = :out, num_nodes = nothing, number_snapshots = nothing, weighted = true)
    s, e, t, val = coo
    if isnothing(number_snapshots)
        num_snapshots = maximum(t)
    else
        num_snapshots = number_snapshots
    end
    if isnothing(num_nodes)
        ns = maximum(s)
        ne = maximum(e)
        num_nodes = max(ns, ne)
    elseif num_nodes isa Integer
        ns = num_nodes
        ne = num_nodes
    elseif num_nodes isa Tuple
        ns = isnothing(num_nodes[1]) ? maximum(s) : num_nodes[1]
        ne = isnothing(num_nodes[2]) ? maximum(e) : num_nodes[2]
        num_nodes = (ns, ne)
    else
        error("Invalid num_nodes $num_nodes")
    end
    @assert isnothing(val) || length(val) == length(s)
    @assert length(s) == length(e) == length(t)
    if !isempty(s)
        @assert minimum(s) >= 1
        @assert minimum(e) >= 1
        @assert maximum(s) <= ns
        @assert maximum(e) <= ne
    end
    num_edges = length(s)
    if !weighted
        coo = (s, e, t, nothing)
    end
    return coo, num_nodes, num_edges, num_snapshots
end