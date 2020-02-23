module Generative

using Flux, Distributions
using Flux: glorot_uniform, @functor
import Distributions: logpdf

include("vae.jl")
include("data_loader/data_loader.jl")

end # module
