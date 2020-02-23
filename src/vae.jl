export VAE_Encoder, kl_divergence, bernoulli_expected_loglik, normal_expected_loglik 
export mean, hidden, log_std


##################
# VAE_Encoder
##################
struct VAE_Encoder{T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
	Wh::MT
	bh::VT
	Wμ::MT
	bμ::VT
	Wlogσ::MT
	blogσ::VT
	f::Function
end
Flux.@functor VAE_Encoder
function VAE_Encoder(in::Integer, h_dim::Integer, out::Integer,
								f=identity; initW=glorot_uniform, initb=zeros)
	Wh = initW(h_dim, in)
	bh = initb(Float32, h_dim)
	Wμ = initW(out, h_dim)
	bμ = initb(Float32, out)
	Wlogσ = initW(out, h_dim)
	blogσ = initb(Float32, out)
	VAE_Encoder(Wh, bh, Wμ, bμ, Wlogσ, blogσ, f)
end

Base.eltype(::Type{<:VAE_Encoder{T}}) where {T} = T
hidden(d::VAE_Encoder, x) = d.f.(d.Wh*x .+ d.bh)
mean(d::VAE_Encoder, h) = d.Wμ*h .+ d.bμ
log_std(d::VAE_Encoder, h) = d.Wlogσ*h .+ d.blogσ

function (d::VAE_Encoder)(x)
	h = hidden(d, x)
	μ = mean(d, h)
	logσ = log_std(d, h)
	z = μ .+ exp.(logσ).*randn(eltype(d))
	z
end

function Base.show(io::IO, en::VAE_Encoder)
	print(io, "VAE(", size(en.Wh, 2), ", ", size(en.Wh, 1), ", ", size(en.Wμ, 1))
	en.f == identity || print(io, ", ", en.f)
	print(io, ")")
end

#############
# KL
#############
function kl_divergence(d::VAE_Encoder, x)
	h = hidden(d, x)
	μ = mean(d, h)
	logσ = log_std(d, h)
	kl = sum(logσ .+ 0.5*μ.^2 .+ 0.5*exp.(2.0*logσ) .- 1)
	kl
end

#####################
# expected log lik
#####################
function bernoulli_expected_loglik(fp, en::VAE_Encoder, x, sample_z=1)
	loglik = 0.0f0
	for i in 1:sample_z
		z = en(x)
		p = fp(z)
		loglik += sum(logpdf.(Bernoulli.(p), x))
	end
	loglik / sample_z
end

function normal_expected_loglik(fμ, flogσ, en::VAE_Encoder, x, sample_z=1)
	loglik = 0.0f0
	for i in 1:sample_z
		z = en(x)
		μ = fμ(z)
		logσ = flogσ(z)
		σ = exp.(logσ)
		loglik += sum(logpdf.(Normal.(μ, σ), x))
	end
	loglik / sample_z
end

