const COOT_T = Tuple{T, T, T, V} where {T <: AbstractVector{<:Integer}, V}
const ADJLIST_T = AbstractVector{T} where {T <: AbstractVector{<:Integer}}
const ADJMAT_T = AbstractArray
