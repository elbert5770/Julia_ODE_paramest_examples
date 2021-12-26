using Plots
using DifferentialEquations
using DataInterpolations
using DiffEqFlux, OrdinaryDiffEq, Flux
using DiffEqSensitivity
using Zygote
using ForwardDiff
using ModelingToolkit
using GalacticOptim
using Profile
using BenchmarkTools

function ODE_FF_eqtns_outer(fixed_params,interp_linear)
	function ODE_FF_eqtns_inner(du,u,p,t)
		alpha,beta = p
		FF = interp_linear(t)
	    du[1] =  FF - alpha*u[1]*u[3]
	    du[2] = alpha*u[1] - beta*u[2]
		du[3] = fixed_params.gamma
	    return nothing
	end
end

function loss_ODE_FF_outer(prob,solution_true,Tdata)
	function loss_ODE_FF_inner(θ)
		sol = Array(solve(remake(prob, p=θ),saveat = Tdata))
		@show θ
		loss = sum(abs2,sol[2,:]-solution_true)
	end
end

function main()
	Tdata = collect(0.0:5.0)
	Mdata = [0.0, 0.03, 0.04, 0.02, 0.01, 0.005]

	interp_linear = DataInterpolations.LinearInterpolation(Mdata, Tdata)
	tspan = (0.0,5.0)
	u0 = [0.0,0.0,1.0]
	p = [1.0,10.0]

	fixed_params = Fixed_params(-0.01)
	ODE_FF_eqtns = ODE_FF_eqtns_outer(fixed_params,interp_linear)
	prob = ODEProblem(ODE_FF_eqtns,u0,tspan,p)
	sol = solve(prob, AutoTsit5(Rosenbrock23(autodiff=true)),saveat=Tdata)
	@show sol(1.0)
	u0 = sol(1.0)
	solution_true = sol[2,:]
	display(plot(sol))

	du = similar(u0)

	ODE_FF_eqtns(du,u0,p,1.0)
	@show du

	p0 = [5.0,5.0]
	loss_ODE_FF = loss_ODE_FF_outer(prob,solution_true,Tdata)
	@show grad = ForwardDiff.gradient(x -> first(loss_ODE_FF(x)), p0)
	@show @btime ForwardDiff.gradient(x -> first($loss_ODE_FF(x)), $p0)
	@show gradz = Zygote.gradient(x -> first(loss_ODE_FF(x)), p0)
	@show @btime Zygote.gradient(x -> first($loss_ODE_FF(x)), $p0)

	@show fdtime = @elapsed grad = ForwardDiff.gradient(x -> first(loss_ODE_FF(x)), p0)
	@show ztime = @elapsed Zygote.gradient(x -> first(loss_ODE_FF(x)), p0)
	result = DiffEqFlux.sciml_train(loss_ODE_FF, p0)

	sol_final = solve(remake(prob,p=result.u))

	p1 = plot(sol_final.t,sol_final[2,:])
	scatter!(p1,sol.t,solution_true)
	display(plot(p1))

end

struct Fixed_params
	gamma::Float64
end
main()
