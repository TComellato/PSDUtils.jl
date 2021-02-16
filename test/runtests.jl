using Test, PSDUtils, LegendTextIO, MJDSigGen, DSP, Distributions
using Random: seed!, MersenneTwister

@testset "PSDUtils.jl" begin
	@testset "todetcoords" begin
		x, y, z = rand(3) .* 50 .- 25
		zlen = rand() * 100

		@test todetcoords(x, y, z, zlen) == ((z + 200) * 10, x * 10, -10 * y + zlen / 2)

		@test todetcoords((x, y, z), zlen) == todetcoords(x, y, z, zlen)
		
		@test todetcoords(0, 5, -200, 100) == (0, 0, 0)

		event = read(DarioHitsFile("test.root.hits"))
		xtal_length = SigGenSetup("GWD6022_01ns.config").xtal_length

		pos = todetcoords.(event.pos, xtal_length)
		todetcoords!(event, xtal_length)

		@test event.pos == pos
	end

	@testset "get_signal!" begin
		setup = SigGenSetup("GWD6022_01ns.config")
		event = rand(collect(DarioHitsFile("test.root.hits")))

		signal = get_signal!(setup, event.pos, event.E)
		signal2 = get_signal!(setup, event)

		@test signal == signal2

		E = sum(event.E)

		@test all(-0.01 * E .< signal .< 1.01 * E)

		@test all(-0.01 * E .< diff(signal))

		@test signal[end] ≈ E

		@test abs(signal[1]) < 0.05 * E
	end

	@testset "charge_cloud_size" begin
		@test charge_cloud_size(1500) == -0.03499440089585633 + 0.0003359462486002238 * 1500

		@test charge_cloud_size(0) == 0.01
	end

	@testset "Signal Processing" begin
		signal = cumsum(rand(100))

		@test getA(signal) == maximum(diff(signal))

		let idx1 = rand(1:50), idx2 = rand(51:100), Δ = idx2 - idx1
			@test get_rise_time(signal, signal[idx1], signal[idx2]) == Δ

			@test get_rise_time(signal, signal[idx1], signal[idx2], 5.3) == 5.3Δ
		end

		@test get_rise_time(signal, signal[9], 101) == (100 + 1) - 9

		@test all(moving_average(signal, 5)[1:3] .≈ sum(signal[1:5]) / 5)
		@test moving_average(signal, 5)[79] ≈ sum(signal[79-2:79+2]) / 5
		@test all(moving_average(signal, 5)[99:100] .≈ sum(signal[96:100]) / 5)

		@test moving_average(signal, 9, 2) == moving_average(moving_average(signal, 9), 9)
	end

	@testset "apply_coll_effects" begin
		w = PSDUtils.gausswindow(0.53, 7, 0.13)

		@test w[1] == w[end] == exp(-0.5 * 7^2)

		@test w[cld(length(w), 2)] == maximum(w)

		@test length(w) == round(Int, 2 * 7 * 0.53 / 0.13)

		signal = cumsum(rand(1000))

		pad = zeros(Float64, 1099)

		PSDUtils.pad_signal!(pad, signal, 100)

		@test pad[1:1000] == signal

		@test all(pad[1001:1099] .== signal[end])

		padcoll   = apply_coll_effects(signal, 0.59, 0.1,  true; nσ = 5)
		nopadcoll = apply_coll_effects(signal, 0.59, 0.1, false; nσ = 5)

		wcoll = PSDUtils.gausswindow(0.59 / (2 * √(2 * log(2))), 5, 0.1)

		@test conv(signal, wcoll) == nopadcoll

		wlen = length(wcoll)
		padsignal = PSDUtils.pad_signal!(zeros(Float64, wlen + 1000 - 1), signal, wlen)

		@test conv(padsignal, wcoll)[1:1000 + wlen - 1] == padcoll
	end

	@testset "noise" begin
		let mt1 = MersenneTwister(123), mt2 = MersenneTwister(123)
			get_noisy_energy(mt1, 1500, 3) == rand(mt2, Normal(1500, 3))
		end

		let mt = MersenneTwister(538)
			seed!(538)
			get_noisy_energy(1500, 3) == rand(mt, Normal(1500, 3))
		end

		noise = randn(1000)

		let signal = cumsum(rand(100)), signal2 = copy(signal)
			addnoise!(signal, noise, 57)
			@test signal == signal2 .+ noise[57:57+99]
		end

		let signal = cumsum(rand(100)), signal2 = copy(signal)
			addnoise!(signal, noise, 970)
			@test signal == signal2 .+ [noise[970:end]; noise[1:69]]
		end

		let signal = cumsum(rand(100)), signal2 = copy(signal)
			addnoise!(MersenneTwister(11235), signal, noise)
			addnoise!(signal2, noise, rand(MersenneTwister(11235), 1:1000))
			@test signal == signal2
		end

		let signal = cumsum(rand(100)), signal2 = copy(signal)
			seed!(23579)
			addnoise!(signal, noise)
			addnoise!(signal2, noise, rand(MersenneTwister(23579), 1:1000))
			@test signal == signal2
		end

		@test_throws BoundsError addnoise!(rand(100), rand(1000), 9970)
	end
end
