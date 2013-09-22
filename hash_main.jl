#!/usr/bin/env julia

import ArgParse
import JSON

import Hash

function parse_args()
    s = ArgParse.ArgParseSettings()
    @ArgParse.add_arg_table s begin
        "learn"
            help = "path to learning vectors"
        "base"
            help = "path to base vectors"
        "query"
            help = "path to query vectors"
        "groundtruth"
            help = "path to groundtruth vectors"
        "--rank", "-r"
            help = "rank of PCA"
            arg_type = Int
            default = 64
        "--neighbors", "-k"
            help = "k of k-NN"
            arg_type = Int
            default = 100
        "--method", "-m"
            help = "hash method (pca, rpca or itq)"
            default = "pca"
    end
    ArgParse.parse_args(s)
end

function run_pcah(learn_fn::String, base_fn::String, query_fn::String,
    groundtruth_fn::String, rank::Int, k::Int, method::String)

    learn = Hash.read_sift_dataset(Float32, learn_fn)
    base = Hash.read_sift_dataset(Float32, base_fn)
    query = Hash.read_sift_dataset(Float32, query_fn)
    groundtruth = Hash.read_sift_dataset(Int32, groundtruth_fn)
    groundtruth += 1  # Make it one-origin
    println(STDERR, "loaded datasets")

    time_to_learn = @elapsed begin
        mu = mean(learn, 2)
        broadcast!(-, learn, learn, mu)

        if method == "pca"
            W, () = Hash.pca(learn, rank)
        elseif method == "rpca"
            W, () = Hash.pca(learn, rank)
            W *= Hash.randortho(Float32, rank)
        elseif method == "itq"
            W, () = Hash.itq(learn, rank)
        elseif method == "random"
            W = randn(size(learn, 1), rank)
        else
            error("invalid method $method")
        end
    end
    println(STDERR, "learn: $time_to_learn sec")

    time_to_embed = @elapsed begin
        broadcast!(-, base, base, mu)
        H = (W' * base) .> 0
    end
    println(STDERR, "embed: $time_to_embed sec")

    time_to_query = @elapsed begin
        broadcast!(-, query, query, mu)
        Hq = (W' * query) .> 0
        result = Hash.scan_hash(Hq, H, k)
    end
    println(STDERR, "query: $time_to_query sec")

    JSON.print(Hash.evaluate(result, groundtruth))
    println(STDERR)
end

args = parse_args()
run_pcah(args["learn"], args["base"], args["query"], args["groundtruth"],
    args["rank"], args["neighbors"], args["method"])
