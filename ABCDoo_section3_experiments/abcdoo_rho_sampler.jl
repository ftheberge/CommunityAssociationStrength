using Pkg
using ABCDGraphGenerator
using Random


# Universal params
seed = 42
d_max_iter = 1000
c_max_iter = 1000
Random.seed!(seed)


# Youtube Graph - no outliers
youtube_params = Dict(
    "name" => "youtube",
    "n" => 52675,
    "nout" => 0,
    "η" => 2.4528144280968203,
    "d_min" => 5,
    "d_max" => 1928,
    "τ₁" => 1.8702187087097446,
    "c_min" => 10,
    "c_max" => 3001,
    "τ₂" => 2.130965769664415,
    "ξ" => 0.5928066048845747,
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
)


for params in [youtube_params, dblp_params, amazon_params]
    for d in [2,5,10]
        for ρ in range(-0.5,0.5,0.1)
            @info "$(params["name"]), d=$d, rho=$ρ"

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

            #@info "Expected value of degree: $(ABCDGraphGenerator.get_ev(τ₁, d_min, d_max))"
            degs = ABCDGraphGenerator.sample_degrees(τ₁, d_min, d_max, n, d_max_iter)

            #@info "Expected value of community size: $(ABCDGraphGenerator.get_ev(τ₂, c_min, c_max))"
            coms = ABCDGraphGenerator.sample_communities(τ₂, ceil(Int, c_min / η), floor(Int, c_max / η), n-nout, c_max_iter)
            pushfirst!(coms, nout)

            #@info "    Done degs and coms, generating graph."
            p = ABCDGraphGenerator.ABCDParams(degs, coms, ξ, η, d, ρ)
            edges, clusters = ABCDGraphGenerator.gen_graph(p)
            open("data/abcdoo_$(name)_d$(d)_rho$(ρ)_edge.dat", "w") do io
                for (a, b) in sort!(collect(edges))
                    println(io, a, "\t", b)
                end
            end
            open("data/abcdoo_$(name)_d$(d)_rho$(ρ)_com.dat", "w") do io
                for (i, c) in enumerate(clusters)
                    println(io, i, "\t", c)
                end
            end
        end
    end
end
