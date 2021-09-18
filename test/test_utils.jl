using ChainRulesTestUtils, FiniteDifferences, Zygote

const rule_config = Zygote.ZygoteRuleConfig()

# Using this until https://github.com/JuliaDiff/FiniteDifferences.jl/issues/188
# is fixed
function FiniteDifferences.to_vec(x::Integer)
    Integer_from_vec(v) = x
    return Int[x], Integer_from_vec
end

function gradtest(l, g::GNNGraph; atol=1e-7, rtol=1e-5,
                                 exclude_grad_fields=[],
                                 broken_grad_fields=[]
                                )
    # TODO these give errors, probably some bugs in ChainRulesTestUtils
    # test_rrule(rule_config, x -> l(g, x), x; rrule_f=rrule_via_ad, check_inferred=false)
    # test_rrule(rule_config, l -> l(g, x), l; rrule_f=rrule_via_ad, check_inferred=false)

    !haskey(g.ndata, :x) && error("Plese pass input graph with :x ndata")
    fdm = central_fdm(5, 1)
    
    x = g.ndata.x
    
    # TEST OUTPUT
    y = l(g, x)
    @test eltype(y) == eltype(x)
    
    g′ = l(g)
    @test g′.ndata.x ≈ y
    
    # TEST INPUT GRADIENT
    x̄  = gradient(x -> sum(l(g, x)), x)[1]
    x̄_fd = FiniteDifferences.grad(fdm, x -> sum(l(g, x)), x)[1]
    @test x̄ ≈ x̄_fd    atol=atol rtol=rtol

    # TEST LAYER GRADIENT - l(g, x) 
    l̄ = gradient(l -> sum(l(g, x)), l)[1]
    l̄_fd = FiniteDifferences.grad(fdm, l -> sum(l(g, x)), l)[1]
    test_approx_structs(l, l̄, l̄_fd; atol, rtol, broken_grad_fields, exclude_grad_fields)
    # TEST LAYER GRADIENT - l(g)
    l̄ = gradient(l -> sum(l(g).ndata.x), l)[1]
    l̄_fd = FiniteDifferences.grad(fdm, l -> sum(l(g).ndata.x), l)[1]
    test_approx_structs(l, l̄, l̄_fd; atol, rtol, broken_grad_fields, exclude_grad_fields)
end

function test_approx_structs(l, l̄, l̄_fd; atol=1e-5, rtol=1e-5, 
            broken_grad_fields=[],
            exclude_grad_fields=[])
    for f in fieldnames(typeof(l))
        f ∈ exclude_grad_fields && continue
        f̄, f̄_fd = getfield(l̄, f), getfield(l̄_fd, f)
        if isnothing(f̄)
            # @show f f̄_fd
            @test !(f̄_fd isa AbstractArray) || isapprox(f̄_fd, fill!(similar(f̄_fd), 0); atol=atol, rtol=rtol)
        elseif f̄ isa Union{AbstractArray, Number}
            @test eltype(f̄) == eltype(getfield(l, f))
            if f ∈ broken_grad_fields
                @test_broken f̄ ≈ f̄_fd   atol=atol rtol=rtol
            else
                # @show f getfield(l, f) f̄ f̄_fd broken_grad_fields
                @test f̄ ≈ f̄_fd   atol=atol rtol=rtol
            end
        else
            test_approx_structs(getfield(l, f), f̄, f̄_fd; broken_grad_fields)
        end
    end
    return true
end


# function gpu_gradtest(l, x_cpu = nothing, args...; test_cpu = true)
#     isnothing(x_cpu) && error("Missing input to test the layers against.")
#     @testset "$name GPU grad tests" begin
#       for layer in layers
#         @testset "$layer Layer GPU grad test" begin
  
#           # compute output and grad of parameters
#           l_cpu = layer(args...)
#           ps_cpu = Flux.params(l_cpu)
#           y_cpu, back_cpu = pullback(() -> sum(l_cpu(x_cpu)), ps_cpu)
#           gs_cpu = back_cpu(1f0)
  
#           x_gpu = gpu(x_cpu)
#           l_gpu = l_cpu |> gpu
#           ps_gpu = Flux.params(l_gpu)
  
#           if typeof(l_gpu) <: BROKEN_LAYERS
#             @test_broken gradient(() -> sum(l_gpu(x_gpu)), ps_gpu) isa Flux.Zygote.Grads
#           else
#             y_gpu, back_gpu = pullback(() -> sum(l_gpu(x_gpu)), ps_gpu)
#             gs_gpu = back_gpu(1f0) # TODO many layers error out when backprop int 1, should fix
  
#             # compute grad of input
#             xg_cpu = gradient(x -> sum(l_cpu(x)), x_cpu)[1]
#             xg_gpu = gradient(x -> sum(l_gpu(x)), x_gpu)[1]
  
#             # test 
#             if test_cpu
#               @test y_gpu ≈ y_cpu rtol=1f-3 atol=1f-3
#               if isnothing(xg_cpu)
#                 @test isnothing(xg_gpu)
#               else
#                 @test Array(xg_gpu) ≈ xg_cpu rtol=1f-3 atol=1f-3
#               end
#             end
#             @test gs_gpu isa Flux.Zygote.Grads
#             for (p_cpu, p_gpu) in zip(ps_cpu, ps_gpu)
#               if isnothing(gs_cpu[p_cpu])
#                 @test isnothing(gs_gpu[p_gpu])
#               else
#                 @test gs_gpu[p_gpu] isa Flux.CUDA.CuArray
#                 if test_cpu
#                   @test Array(gs_gpu[p_gpu]) ≈ gs_cpu[p_cpu] rtol=1f-3 atol=1f-3
#                 end
#               end
#             end
#           end
#         end
#       end
#     end
#   end