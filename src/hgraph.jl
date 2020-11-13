struct HierarchicalGraph{T<:Integer}
    layers :: Vector{SimpleGraph{T}}
end
HierarchicalGraph{T}() where {T} = HierarchicalGraph{T}(Vector{SimpleGraph{T}}())
HierarchicalGraph() = HierarchicalGraph{Int}()

layers(g) = g.layers

layer(g, l) = layers(g)[l]

nlayers(g::HierarchicalGraph) = length(layers(g))

add_layer!(g::HierarchicalGraph{T}) where {T} = push!(layers(g), SimpleGraph{T}(nv(g)))

add_layers!(g::HierarchicalGraph, n) =
    for i in 1:n
        add_layer!(g)
    end

LightGraphs.add_vertex!(g::HierarchicalGraph) =
    for gg in layers(g)
        add_vertex!(gg)
    end

LightGraphs.add_vertices!(g::HierarchicalGraph, n) =
    for gg in layers(g)
        add_vertices!(g, n)
    end

LightGraphs.nv(g::HierarchicalGraph{T}) where {T} = iszero(nlayers(g)) ? zero(T) : nv(layer(g, 1))

LightGraphs.ne(g::HierarchicalGraph, l) = ne(layer(g, l))
LightGraphs.ne(g::HierarchicalGraph) = iszero(nlayers(g)) ? 0 : sum(ne, layers(g))

function LightGraphs.add_edge!(g::HierarchicalGraph{T}, _l, _x, _y) where {T}
    # ensure there are enough layers
    L = nlayers(g)
    l = oftype(L, _l)
    if l > L
        add_layers!(g, l - L)
    end
    # ensure there are enough vertices
    n = nv(g)
    x = oftype(n, _x)
    y = oftype(n, _y)
    z = max(x, y)
    if z > n
        add_vertices!(gg, z - n)
    end
    # add the edge
    add_edge!(layer(g, l), x, y)
end

LightGraphs.neighbors(g::HierarchicalGraph, l, x) = neighbors(layer(g, l), x)
