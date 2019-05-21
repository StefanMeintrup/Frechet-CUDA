#pragma once

#include <boost/chrono/include.hpp>

#include "geometry_basics.hpp"
#include "curve.hpp"
#include "intersection_algorithm_in_parallel.hpp"

namespace Frechet {
namespace Continuous {
	
	struct Result {
		distance_t value;
		double time_searches;
		double time_bounds;
		std::size_t number_searches;
	};
	
	auto distance(const Curve&, const Curve&, distance_t, distance_t, 
			const distance_t = std::numeric_limits<distance_t>::epsilon(), bool = true) -> Result;
	auto distance_cuda(const Curve&, const Curve&, distance_t, distance_t, 
			const distance_t = std::numeric_limits<distance_t>::epsilon(), bool = true) -> Result;
	bool _lessThan(const distance_t, const Curve&, const Curve&, 
			std::vector<std::vector<distance_t>>&, std::vector<std::vector<distance_t>>&);
	bool _lessThan_cuda(
		const distance_t, 
		const Curve&, 
		const Curve&, 
		std::vector<std::vector<distance_t>>&, 
		std::vector<std::vector<distance_t>>&,
		Cuda_intersection&);
}
namespace Discrete {
	
	struct Result {
		distance_t value;
		double time;
	};
	
	auto distance(const Curve&, const Curve&) -> Result;
	auto _dp(std::vector<std::vector<distance_t>> &a, const std::size_t i, const std::size_t j, 
			const Curve &curve1, const Curve &curve2) -> distance_t;
}
}
