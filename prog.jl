using LinearAlgebra

setprecision(32)

function randSquareMatrix(n::Int64)
    return BigFloat.(rand(n, n))
end

function testMethod(f::Function; tests=100::Int64, n=2:10, norma=(x -> norm(x, 2)), log=true)
    results_mean = BigFloat.(zeros(length(n)))
    results_max = BigFloat.(zeros(length(n)))
    Σ = BigFloat(0)
    i = 1
    for size = n
        for t = 1:tests
            m = randSquareMatrix(size)
            res = norma(m * f(m) - BigFloat.(eye(size)))
            Σ += res
            if res > results_max[i]
                results_max[i] = res
            end
        end
        results_mean[i] = Σ / tests
        if log
            @printf "Matrix size: %3d Tests: %5d Mean error: %.4e Max error: %.4e" size tests results_mean[i] results_max[i]
        end
        i += 1
    end
    return results_mean, results_max
end

function LUDecomposition(m::Matrix{BigFloat})
    n = size(m)[1]
    u = Matrix{BigFloat}(n, n)
    l = Matrix{BigFloat}(n, n)
    u[1, 1] = m[1, 1]
    l[1, 1] = 1
    for j = 2:n
        u[1, j] = m[1, j]
        l[j, 1] = m[j, 1] / u[1, 1]
    end
    for i = 2:n
        for j = 1:i-1
            u[i, j] = 0
            l[j, i] = 0
        end
        u[i, i] = m[i, i] - sum([l[i, k] * u[k, i] for k = 1:i-1])
        l[i, i] = 1
        for j = i+1:n
            u[i, j] = m[i, j] - sum([l[i, k] * u[k, j] for k = 1:i-1])
            l[j, i] = (m[j, i] - sum([l[j, k] * u[k, i] for k = 1:i-1])) / u[i, i]
        end
    end
    return l, u
end

function Gauss(A::Matrix{BigFloat})
    n = size(A)[1]
    v = BigFloat.(eye(n))
    m = hcat(A, v)
    for i = 1:n
        p = sortperm(abs.(m[i:end, i]), rev=true) + i - 1
        m[i:end, :] = m[p, :]
        m[i, :] /= m[i, i]
        for j = i+1:n
            a = m[j, i]
            b = m[i, i]
            m[j, :] *= b
            m[j, :] -= a * m[i, :]
            m[j, :] /= b
        end
    end
    for i = n:-1:2
        for j = 1:i-1
            a = m[j, i]
            m[j, :] -= a * m[i, :]
        end
    end
    return m[:, n+1:end]
end

function makeInverse(f::Function, A::Matrix{BigFloat}; A⁻¹::Matrix{BigFloat}=zeros(A), max_iter=Inf)
    n = size(A)[1]
    B = BigFloat.(zeros(n, n))
    E = BigFloat.(eye(n))
    for i = 1:n
        B[:, i] = f(A, E[:, i], A⁻¹[:, i], iterations=max_iter)
    end
    return B
end 

function makeCleverInverse(f::Function, A::Matrix{BigFloat}; A⁻¹::Matrix{BigFloat}=zeros(A), max_iter=Inf)
    n = size(A)[1]
    B = BigFloat.(zeros(n, n))
    E = copy(A')
    for i = 1:n
        B[:, i] = f(A' * A, E[:, i], A⁻¹[:, i], iterations=max_iter)
    end
    return B
end

function Richardson(A::Matrix{BigFloat}, b::Vector{BigFloat}, x::Vector{BigFloat}; τ=3, iterations=Inf)
    i = 0
    while i < iterations
        δ = - τ * A * x + τ * b
        if norm(δ) == 0
            i = Inf
        end
        x += δ
        i += 1
    end
    return x
end

function SOR(A::Matrix{BigFloat}, b::Vector{BigFloat}, x::Vector{BigFloat}; ω=1, iterations=Inf)
    t = 0
    n = size(A)[1]
    while t < iterations
        change = 0
        for i = 1:n
            σ = 0
            for j = 1:n
                if i != j
                    σ += A[i, j] * x[j]
                end
            end
            δ = ω * ((b[i] - σ) / A[i, i] - x[i])
            x[i] += δ
            change += δ
        end
        if change == 0
            @printf "SOR iterations: %d\n" t
            t = Inf
        end
        t += 1
    end
    return x
end

function test(f::Function, count::Int64, size::Int64)
    max = BigFloat.(zeros(3))
    sum = BigFloat.(zeros(3))
    func_time = Dates.Millisecond(0)
    for i = 1:count
        m = randSquareMatrix(size)
        
        x = now()
        m1 = f(m)
        func_time += now() - x

        res = m * m1 - BigFloat.(eye(size))

        res1 = norm(res, 1)
        res2 = norm(res, 2)
        res∞ = norm(res, Inf)

        if res1 > max[1] max[1] = res1 end
        if res2 > max[2] max[2] = res2 end
        if res∞ > max[3] max[3] = res∞ end

        sum[1] += res1
        sum[2] += res2
        sum[3] += res∞
    end
    print(func_time)
    @printf "\n%4d %3d %.4e %.4e %.4e %.4e %.4e %.4e\n" count size sum[1] / count sum[2] / count sum[3] / count max[1] max[2] max[3]

end

setprecision(64)

test(inv, 1000, 10)
test(Gauss, 1000, 10)
