using Pkg
using ABCDGraphGenerator
using Random

# LiveJournal graph vars
lj_params = Dict(
    "name"  => "lj",
    "n"     => 3997962,
    "nout"  => 2850014,
    "η"     => 6.24,
    "d_min" => 5,
    "d_max" => 14815,
    "τ₁"    => 1.74,
    "c_min" => 10,
    "c_max" => 178899,
    "τ₂"    => 1.88,
    "ξ"     => 0.71,
    "ρ"     => 0.43,
)

# Youtube Graph vars
youtube_params = Dict(
    "name"  => "youtube",
    "n"     => 1134890,
    "nout"  => 1082215,
    "η"     => 2.45,
    "d_min" => 5,
    "d_max" => 28754,
    "τ₁"    => 2.09,       
    "c_min" => 10,
    "c_max" => 3001,
    "τ₂"    => 2.13,
    "ξ"     => 0.96,
    "ρ"     => 0.16,
)

# Amazon Graph vars
amazon_params = Dict(
    "name"  => "amazon",
    "n"     => 334863,
    "nout"  => 17669,
    "η"     => 7.16,
    "d_min" => 5,
    "d_max" => 549,
    "τ₁"    => 3.04,
    "c_min" => 10,
    "c_max" => 53551,
    "τ₂"    => 2.03,
    "ξ"     => 0.11,
    "ρ"     => 0.22,
)

# DBLP Graph vars
dblp_params = Dict(
    "name"  => "dblp",
    "n"     => 317080,
    "nout"  => 56082,
    "η"     => 2.76,
    "d_min" => 5,
    "d_max" => 343,
    "τ₁"    => 2.30,       
    "c_min" => 10,
    "c_max" => 7556,
    "τ₂"    => 1.88,
    "ξ"     => 0.11,
    "ρ"     => 0.76,
)


# Universal params
seed = 42
d_max_iter = 1000
c_max_iter = 1000
Random.seed!(seed)

for params in [dblp_params, amazon_params, lj_params, youtube_params]
    for d in [2,5,10]
        @info "$(params["name"]), d=$d"

        name = params["name"]
        n = params["n"]
        nout = params["nout"]
        η = params["η"]
        d_min = params["d_min"]
        d_max = params["d_max"]
        τ₁ = params["τ₁"]
        c_min = params["c_min"]
        c_max = params["c_max"]
        τ₂ = params["τ₂"]
        ξ = params["ξ"]
        #ρ = params["ρ"]


        # in what follows n is number of non-outlier nodes
        n = n - nout

        # Actually Generate the graph
        #@info "Expected value of degree: $(ABCDGraphGenerator.get_ev(τ₁, d_min, d_max))"
        degs = ABCDGraphGenerator.sample_degrees(τ₁, d_min, d_max, n + nout, d_max_iter)
        #open(io -> foreach(d -> println(io, d), degs), "deg.dat", "w")
        @assert iseven(sum(degs))


        #@info "Expected value of community size: $(ABCDGraphGenerator.get_ev(τ₂, c_min, c_max))"
        coms = ABCDGraphGenerator.sample_communities(τ₂, ceil(Int, c_min / η), floor(Int, c_max / η), n, c_max_iter)
        @assert sum(coms) == n
        pushfirst!(coms, nout)

        @info "    Done degs and coms, generating graph."
        p = ABCDGraphGenerator.ABCDParams(degs, coms, ξ, η, d, ρ)
        edges, clusters = ABCDGraphGenerator.gen_graph(p)
        open("abcdoo_$(name)_d$(d)_edge.dat", "w") do io
        #open("abcdoo_$(name)_d$(d)_nocorr_edge.dat", "w") do io
           for (a, b) in sort!(collect(edges))
               println(io, a, "\t", b)
           end
        end
        open("abcdoo_$(name)_d$(d)_com.dat", "w") do io
        #open("abcdoo_$(name)_d$(d)_nocorr_com.dat", "w") do io
            for (i, c) in enumerate(clusters)
                println(io, i, "\t", c)
            end
        end

    end
end
