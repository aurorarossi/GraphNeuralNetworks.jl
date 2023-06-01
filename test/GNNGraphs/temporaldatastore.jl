@testset "constructor" begin
    @test_throws AssertionError TemporalDataStore(5,3, (:x => rand(10,5,3), :y => rand(3,3,5,2)))
    @test_throws AssertionError TemporalDataStore(5,3, (:x => rand(10,4,3), :y => rand(3,3,5,3)))

    x = rand(10,5)
    @test  TemporalDataStore(5, 1, (:x => x)) == DataStore(5, (:x => x))

    @testset "keyword args" begin
        tds = TemporalDataStore(4, 10, x = rand(2,4,10), y = rand(4, 10))
        @test size(tds.x) == (2, 4, 10)
        @test size(tds.y) == (4, 10)

        tds = TemporalDataStore(x = rand(2,4,10), y = rand(4, 10)) #possible feat: should it understand by itself n and t?
        @test size(tds.x) == (2, 4, 10)
        @test size(tds.y) == (4, 10)
    end
end;

@testset "getdata / getn / gett" begin
    tds = TemporalDataStore(4, 10, x = rand(2,4,10))
    @test getdata(tds) == getfield(tds, :_data)
    @test_throws KeyError tds.data
    @test getn(tds) == getfield(tds, :_n)
    @test_throws KeyError tds.n
    @test gett(tds) == getfield(tds, :_t)
    @test_throws KeyError tds.t
end;

@testset "getproperty / setproperty!" begin
    x = rand(10,5)
    z = rand(3,10,5)
    tds = TemporalDataStore(10, 5, (:x => x))
    @test tds.x == tds[:x] == x
    @test_throws DimensionMismatch tds.z=rand(10,4)
    tds.z = z
    @test tds.z == z
end;

@testset "map" begin
    tds = TemporalDataStore(5, 10, (:x => rand(5,10), :y => rand(2,5, 10)))
    tds2 = map(x -> x .+ 1, tds)
    @test tds2.x == tds.x .+ 1
    @test tds2.y == tds.y .+ 1

    @test_throws AssertionError tds2 = map(x -> [x; x], tds)
end;

@testset "getobs / getsnaps" begin
    x=rand(3,5,10)
    tds = TemporalDataStore(5, 10, (:x => x))
    @test getobs(tds, 1).x == x[:,1,:]
    @test getobs(tds, [1,2]).x == x[:,[1,2],:]
    @test getsnaps(tds, 10).x == x[:,:,end]
    @test getsnaps(tds, [1,2]).x == x[:,:,[1,2]]
end;

@testset "cat" begin
    tds1 = TemporalDataStore(3, 4, x=rand(2,3,4))
    tds2 = TemporalDataStore(2, 4, x=rand(2,2,4))

    tds = GNNGraphs.cat_features(tds1, tds2)
    @test getn(tds) == 5
end;

@testset "gradient" begin
    tds = TemporalDataStore(3, 4, x=rand(2,3,4))

    f1(tds) = sum(tds.x)
    grad = gradient(f1, tds)[1]
    @test grad._data[:x] ≈ ngradient(f1, tds)[1][:x]
end;

@testset "functor" begin
    tds1 = TemporalDataStore(3, 4, x=rand(2,3,4))
    p, re = Functors.functor(tds1)
    @test p[1] === getn(tds1)
    @test p[2] === gett(tds1)
    @test p[3] == getdata(tds1)
    @test tds1 == re(p)

    tds2 = Functors.fmap(tds1) do x
        if x isa AbstractArray
            x .+ 1
        else
            x
        end
    end
    @test tds1 isa TemporalDataStore
    @test tds2.x == tds1.x .+ 1
end;
