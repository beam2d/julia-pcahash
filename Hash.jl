module Hash

export read_sift_dataset, centering!, centering, randortho, pca, itq, scan_hash,
       Evaluation, evaluate

import Base.Collections

# Reads a sift dataset of element-type T (see http://corpus-texmex.irisa.fr/).
function read_sift_dataset{T<:Real}(::Type{T}, file_name::String)
    open(file_name) do file
        num_elements = read(file, Int32)
        num_samples = div(stat(fd(file)).size, 4 + num_elements * sizeof(T))
        seekstart(file)
        X = Array(T, num_elements, num_samples)
        for i=1:num_samples
            skip(file, 4)
            X[:, i] = read(file, T, num_elements)
        end
        X
    end
end

# Creates a random orthogonal matrix.
function randortho{T<:Real}(::Type{T}, n::Int)
    A = convert(Matrix{T}, randn(n, n))
    qrfact!(A)[:Q]
end

# Computes principal components of a column-wise data matrix.
function pca(X::Matrix, rank::Int)
    V, s, U = svd(X')
    return U'[:, 1:rank], broadcast(*, s[1:rank], V[:, 1:rank]')
end

# Iterative Quantization from a zero-centered data matrix.
function itq{T<:Real}(X::Matrix{T}, rank::Int, iterations::Int=50)
    W, V = pca(X, rank)
    R = randortho(T, rank)
    B = ones(T, (rank, size(X, 2)))
    for iter=1:iterations
        B = copysign(B, R * V)
        S, Omega, S_hat = svd(B * V')
        R = S_hat' * S'
    end
    W * R, R * V
end

hamming_distance(v::BitVector, w::BitVector) =
    ccall((:hamming_distance, "./hamming_distance.so"),
        Int32, (Ptr{Uint64}, Ptr{Uint64}, Int), v.chunks, w.chunks, length(v))

# Searches nearest neighbors of given queries using binary hashing.
# NOTE: Returned indexes are one-origin, while groundtruth file is zero-origin.
function scan_hash(Hq::BitMatrix, H::BitMatrix, k::Int)
    neighbors = Array(Int32, (k, size(Hq, 2)))
    nbits = size(Hq, 1)
    hq = BitArray(nbits)
    h = BitArray(nbits)
    for i=1:size(Hq, 2)
        heap = Array((Int32, Int32), 0)
        copy!(hq, 1, Hq, 1 + (i-1)nbits, nbits)
        for j=1:size(H, 2)
            copy!(h, 1, H, 1 + (j-1)nbits, nbits)
            score = hamming_distance(hq, h)
            if length(heap) == k
                if score >= heap[1][1]
                    continue
                end
                Base.Collections.heappop!(heap, Base.Order.Reverse)
            end
            Base.Collections.heappush!(heap, (score, int32(j)), Base.Order.Reverse)
        end
        neighbors[:, i] = [h[2] for h in sort(heap)]
        print(STDERR, "$i / $(size(Hq, 2))\r")
    end
    println(STDERR)
    neighbors
end

type Evaluation
    precision::Float64
    recall::Float64
end

function evaluate{T<:Integer}(neighbors::Matrix{T}, groundtruth::Matrix{T})
    sum_precision = 0.
    sum_recall = 0.
    for i=1:size(neighbors, 2)
        hits = length(intersect(neighbors[:, i], groundtruth[:, i]))
        sum_precision += hits / size(neighbors, 1)
        sum_recall += hits / size(groundtruth, 1)
    end
    Evaluation(sum_precision / size(neighbors, 2), sum_recall / size(neighbors, 2))
end

end  # module Hash
