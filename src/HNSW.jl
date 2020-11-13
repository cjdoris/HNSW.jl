module HNSW

using Random, Distances, DataStructures
using Base: @propagate_inbounds

export HNSWNeighbour, HNSWNode, HNSWLayer, HNSWGraph, HNSWParams, hnsw_search, hnsw_insert, hnsw_insert_many!, hnsw_layers, hnsw_nlayers, hnsw_layer, hnsw_nodes, hnsw_node, hnsw_nnodes, hnsw_values, hnsw_value, hnsw_nbrs, hnsw_degree, hnsw_dist, hnsw_edges

struct HNSWNeighbour
    idx :: Int
    dist :: Float64
end

struct HNSWNode
    nbrs :: Vector{HNSWNeighbour}
    value_idx :: Int
    down_idx :: Int
    up_idx :: Int
end
HNSWNode(value_idx, down_idx, up_idx) = HNSWNode(Vector{HNSWNeighbour}(), value_idx, down_idx, up_idx)

struct HNSWLayer
    nodes :: Vector{HNSWNode}
end
HNSWLayer() = HNSWLayer(Vector{HNSWNode}())

"""
    HNSWGraph([values], [metric=Euclidean()])

Construct a HNSW graph from the given `values` using the given `metric`.
"""
struct HNSWGraph{T, M<:SemiMetric}
    layers :: Vector{HNSWLayer}
    values :: Vector{T}
    metric :: M
end
HNSWGraph{T}(layers, values, metric::M) where {T, M<:SemiMetric} = HNSWGraph{T, M}(layers, values, metric)
HNSWGraph{T}(metric::SemiMetric) where {T} = HNSWGraph{T}(Vector{HNSWLayer}(), Vector{T}(), metric)
HNSWGraph{T}(qs, metric::SemiMetric=Euclidean(); params...) where {T} = hnsw_insert_many!(HNSWGraph{T}(metric), qs; params...)
HNSWGraph(qs, metric::SemiMetric=Euclidean(); params...) = HNSWGraph{eltype(qs)}(qs, metric; params...)

"""
    hnsw_layers(g)

The layers of `g`.
"""
@propagate_inbounds hnsw_layers(g::HNSWGraph) = g.layers

"""
    hnsw_layers(g)

The number of layers in `g`.
"""
@propagate_inbounds hnsw_nlayers(g::HNSWGraph) = length(hnsw_layers(g))

"""
    hnsw_layer(g, L)

The `L`th layer in `g`.
"""
@propagate_inbounds hnsw_layer(g::HNSWGraph, L::Integer) = hnsw_layers(g)[L]

"""
    hnsw_addemptylayer!(g)

Adds an empty layer to `g`.
"""
@propagate_inbounds hnsw_addemptylayer!(g::HNSWGraph) = push!(g.layers, HNSWLayer())

"""
    hnsw_nodes(gL)
    hnsw_nodes(g, L)

The nodes of the `L`th layer of `g`, or layer `gL`.
"""
@propagate_inbounds hnsw_nodes(gL::HNSWLayer) = gL.nodes
@propagate_inbounds hnsw_nodes(g::HNSWGraph, L) = hnsw_nodes(hnsw_layer(g, L))

"""
    hnsw_nnodes(g)

The number of nodes in `g`.
"""
@propagate_inbounds hnsw_nnodes(g::HNSWGraph) = length(g.values)

"""
    hnsw_nnodes(gL)
    hnsw_nnodes(g, L)

The number of nodes in the `L`th layer of `g`, or layer `gL`.
"""
@propagate_inbounds hnsw_nnodes(gL::HNSWLayer) = length(hnsw_nodes(gL))
@propagate_inbounds hnsw_nnodes(g::HNSWGraph, L) = hnsw_nnodes(hnsw_layer(g, L))

"""
    hnsw_values(g)

The values of nodes of `g`.
"""
@propagate_inbounds hnsw_values(g::HNSWGraph) = g.values

"""
    hnsw_value(...)

The value of `hnsw_node(...)`
"""
@propagate_inbounds hnsw_value(g::HNSWGraph, i::Integer) = hnsw_values(g)[i]
@propagate_inbounds hnsw_value(g::HNSWGraph, n::HNSWNode) = hnsw_value(g, n.value_idx)
@propagate_inbounds hnsw_value(g::HNSWGraph, args...) = hnsw_value(g, hnsw_node(g, args...))

"""
    hnsw_node(gL, i)
    hnsw_node(g, L, i)

The `i`th node of the `L`th layer of `g`, or layer `gL`.
"""
@propagate_inbounds hnsw_node(gL::HNSWLayer, i) = gL.nodes[hnsw_idx(i)]
@propagate_inbounds hnsw_node(g::HNSWGraph, L, i) = hnsw_node(hnsw_layer(g, L), i)

"""
    hnsw_newnbr(g, L, i, x)

Construct a new `HNSWNeighbour` from `x` to `hnsw_node(g, L, i)`.
"""
@propagate_inbounds hnsw_newnbr(g::HNSWGraph, L, i, x) = HNSWNeighbour(hnsw_idx(i), hnsw_dist(g, L, i, x))

"""
    hnsw_newnbr(nbr, i)

A new `HNSWNeighbour` with the same distance as `nbr` but index `i`.
"""
@propagate_inbounds hnsw_newnbr(c::HNSWNeighbour, i) = HNSWNeighbour(hnsw_idx(i), c.dist)

"""
    hnsw_downnbr(g, L, nbr)

A neighbour equivalent to `HNSWNeighbour` but in the layer below.
"""
@propagate_inbounds hnsw_downnbr(c::HNSWNeighbour, n::HNSWNode) = hnsw_newnbr(c, n.down_idx)
@propagate_inbounds hnsw_downnbr(g::HNSWGraph, L, c::HNSWNeighbour) = hnsw_downnbr(c, hnsw_node(g, L, c))

"""
    hnsw_nbrs(...)

The neighbours of `hnsw_node(...)`.
"""
@propagate_inbounds hnsw_nbrs(n::HNSWNode) = n.nbrs
@propagate_inbounds hnsw_nbrs(args...) = hnsw_nbrs(hnsw_node(args...))

"""
    hnsw_degree(...)

The degree (number of neighbours) of `hnsw_node(...)`.
"""
@propagate_inbounds hnsw_degree(args...) = length(hnsw_nbrs(args...))

"""
    hnsw_idx(i)

Interpret `i` as an integer index. The argument may be an integer or a `HNSWNeighbour`.
"""
@propagate_inbounds hnsw_idx(i::Integer) = i
@propagate_inbounds hnsw_idx(n::HNSWNeighbour) = n.idx

"""
    hnsw_dist(nbr)
    hnsw_dist(g, L, i, x)

Compute or fetch the distance.
"""
@propagate_inbounds hnsw_dist(n::HNSWNeighbour) = n.dist
@propagate_inbounds hnsw_dist(g::HNSWGraph, L, i, x) = evaluate(g.metric, x, hnsw_value(g, L, i))

"""
    hnsw_edges(gL, i)
    hnsw_edges(g, L, i)

Iterator of edges out of the `i`th node of the `L`th layer of `g`, or of layer `gL`.

Each edge is a 3-tuple `(node1, node2, dist)` where `node1` is the given node.
"""
@propagate_inbounds hnsw_edges(gL::HNSWLayer, i) = (n=hnsw_node(gL, i); ((n, hnsw_node(gL, b), b.dist) for b in hnsw_nbrs(n)))
@propagate_inbounds hnsw_edges(g::HNSWGraph, L, i) = hnsw_edges(hnsw_layer(g, L), i)

"""
    hnsw_edges(gL)
    hnsw_edges(g, L)

Iterator of edges in the `L`th layer of `g`, or layer `gL`.

Each edge is a 3-tuple `(node1, node2, dist)`.
"""
@propagate_inbounds hnsw_edges(gL::HNSWLayer) = ((x,y,d) for i in 1:hnsw_nnodes(gL) for (x,y,d) in hnsw_edges(gL, i) if x.value_idx < y.value_idx)
@propagate_inbounds hnsw_edges(g::HNSWGraph, L) = hnsw_edges(hnsw_layer(g, L))

"""
    hnsw_edges(g)

Iterator of edges in `g`.

Each edge is a 4-tuple `(node1, node2, dist, layer)`.
"""
@propagate_inbounds hnsw_edges(g::HNSWGraph) = ((x..., L) for L in 1:hnsw_nlayers(g) for x in hnsw_edges(g, L))

"""
    hnsw_search(g::HNSWGraph, q, k::Integer; poolsize=10) :: Vector{HNSWNeighbour}

Find the `k` nearest neighbours of `q` in `g`.

Output is sorted by distance.
"""
function hnsw_search(g::HNSWGraph{T}, q, k::Integer; poolsize::Integer=10) where {T}
    L0 = hnsw_nlayers(g)
    pool = [hnsw_newnbr(g, L0, 1, q)]
    for L in L0:-1:1
        cands = hnsw_search_layer(g, L, pool, q, L==1 ? max(poolsize, k) : poolsize)
        if L == 1
            return cands[1:k]
        else
            pool = [hnsw_downnbr(g, L, c) for c in cands]
        end
    end
    # we get here only if L0=0, i.e. the graph is empty
    return Vector{HNSWNeighbour}()
end

"""
    hnsw_search_layer(g::HNSWLayer, L::Integer, pool::Vector{Int}, q, k::Integer) :: Vector{HNSWNeighbour}

Find the `k` nearest neighbours of `q` in layer `L` of `g` starting from `pool`.

The pool must be sorted by distance to `q`. Output is sorted by distance.
"""
function hnsw_search_layer(g::HNSWGraph, L::Integer, pool::Vector{HNSWNeighbour}, q, k::Integer)
    visited = Set(p.idx for p in pool)
    cands = pool[1:min(end,k)]
    result = copy(cands)
    while !isempty(cands)
        cand = popfirst!(cands)
        cand.dist > last(result).dist && break
        for nbr in hnsw_node(g, L, cand).nbrs
            nbr.idx ∈ visited && continue
            push!(visited, nbr.idx)
            cand2 = hnsw_newnbr(g, L, nbr.idx, q)
            if cand2.dist < last(result).dist || length(result) < k
                insert!(cands, searchsortedfirst(cands, cand2, by=x->x.dist), cand2)
                insert!(result, searchsortedfirst(result, cand2, by=x->x.dist), cand2)
                length(result) > k && pop!(result)
            end
        end
    end
    return result
end

"""
    hnsw_insert(g::HNSWGraph, q; poolsize=100, degree=10, degree0=degree, meanlayers=1.0)

Insert vector `q` into `g`.
"""
function hnsw_insert!(g::HNSWGraph{T}, _q; poolsize::Int=100, degree::Int=10, degree0::Int=degree, meanlayers::Real=1.0) where {T}
    q = convert(T, _q)

    # select the layer in which to insert q, and ensure g contains this many layers
    L0 = hnsw_nlayers(g)
    Lq = 1 + floor(Int, meanlayers * randexp())
    while hnsw_nlayers(g) < Lq
        hnsw_addemptylayer!(g)
    end

    # layers above Lq: get an entry point
    pool = L0==0 ? HNSWNeighbour[] : [hnsw_newnbr(g, L0, 1, q)]
    for L in L0:-1:Lq+1
        cands = hnsw_search_layer(g, L, pool, q, 1)
        pool = [hnsw_downnbr(g, L, c) for c in cands]
    end

    # layers above L0: insert isolated node
    push!(g.values, q)
    value_idx = length(g.values)
    up_idx = 0
    for L in Lq:-1:L0+1
        node = HNSWNode(value_idx, L==1 ? 0 : hnsw_nnodes(g, L-1)+1, up_idx)
        push!(g.layers[L].nodes, node)
        up_idx = hnsw_nnodes(g, L)
    end

    # remaining layers
    for L in min(L0,Lq):-1:1
        # candidates in this layer
        cands = hnsw_search_layer(g, L, pool, q, poolsize)
        # neighbours
        k = L==1 ? degree0 : degree
        nbrs = cands[1:min(end,k)]
        # insert node with these neighbours
        node = HNSWNode(nbrs, value_idx, L==1 ? 0 : hnsw_nnodes(g, L-1)+1, up_idx)
        push!(g.layers[L].nodes, node)
        idx = up_idx = hnsw_nnodes(g, L)
        # insert reverse neighbours
        for nbr in nbrs
            node2 = hnsw_node(g, L, nbr)
            nbr2 = HNSWNeighbour(idx, nbr.dist)
            i = searchsortedfirst(node2.nbrs, nbr2, by=x->x.dist)
            insert!(node2.nbrs, i, nbr2)
            # if there are too many, delete some
            if length(node2.nbrs) > k
                @assert length(node2.nbrs) == k+1
                nbr3 = pop!(node2.nbrs)
                node3 = hnsw_node(g, L, nbr3)
                for (j,nbr4) in enumerate(node3.nbrs)
                    if nbr4.idx == nbr.idx
                        deleteat!(node3.nbrs, j)
                        break
                    end
                end
            end
        end
        # the pool in the layer below
        pool = [hnsw_downnbr(g, L, c) for c in cands]
    end

    return g
end

"""
    hnsw_insert_many!(g::HNSWGraph, qs; ...)

Insert each element of `qs` into `g`. Parameters are as for `hnsw_insert!`.
"""
function hnsw_insert_many!(g::HNSWGraph, qs; params...)
    for q in qs
        hnsw_insert!(g, q; params...)
    end
    return g
end

end # module
