#include <vector>
#include <limits>

#include <boost/chrono/include.hpp>

#include "frechet.hpp"
#include "intersection_algorithm_in_parallel.hpp"

namespace Frechet {

namespace Continuous {

auto distance_cuda(const Curve &curve1, const Curve &curve2, distance_t ub, distance_t lb, const distance_t eps, bool benchmark) -> Result {
	Result result;
	auto start = boost::chrono::process_real_cpu_clock::now();
	
	distance_t split = (ub + lb)/2;
	std::size_t number_searches = 0;
	
	if (benchmark) {

		if (ub - lb > eps) {
			auto infty = std::numeric_limits<distance_t>::infinity();
			std::vector<std::vector<distance_t>> reachable1(curve1.size()-1, std::vector<distance_t>(curve2.size(), infty));
			std::vector<std::vector<distance_t>> reachable2(curve1.size(), std::vector<distance_t>(curve2.size()-1, infty));
			
			distance_t *host_results_p = static_cast<distance_t*>(malloc(2 * (2 * curve1.size() * curve2.size()) * sizeof(distance_t)));
			Cuda_intersection cuda = Cuda_intersection(curve1, curve2, host_results_p);

			//Binary search over the feasible distances
			while (ub - lb > eps) {
				++number_searches;
				split = (ub + lb)/2;
				auto isLessThan = false;
				{					
					isLessThan = _lessThan_cuda(split, curve1, curve2, reachable1, reachable2, cuda);
				}
				if (isLessThan) {
					ub = split;
				}
				else {
					lb = split;
				}
				#if DEBUG
				std::cout << "narrowed to [" << lb << ", " << ub << "]" << std::endl;
				#endif
			}
			cuda.free_memory();
			free(host_results_p);
		}	
	} else {
		if (ub - lb > eps) {
			auto infty = std::numeric_limits<distance_t>::infinity();
			std::vector<std::vector<distance_t>> reachable1(curve1.size()-1, std::vector<distance_t>(curve2.size(), infty));
			std::vector<std::vector<distance_t>> reachable2(curve1.size(), std::vector<distance_t>(curve2.size()-1, infty));
			
			distance_t *host_results_p = static_cast<distance_t*>(malloc(2 * (2 * curve1.size() * curve2.size()) * sizeof(distance_t)));
			Cuda_intersection cuda = Cuda_intersection(curve1, curve2, host_results_p);
	
			//Binary search over the feasible distances
			while (ub - lb > eps) {
				++number_searches;
				split = (ub + lb)/2;
				auto isLessThan = _lessThan_cuda(split, curve1, curve2, reachable1, reachable2, cuda);
				if (isLessThan) {
					ub = split;
				}
				else {
					lb = split;
				}
				#if DEBUG
				std::cout << "narrowed to [" << lb << ", " << ub << "]" << std::endl;
				#endif
			}
			cuda.free_memory();
			free(host_results_p);
		}
	}
	auto value = std::round((ub + lb)/2.*1000)/1000;
	auto end = boost::chrono::process_real_cpu_clock::now();
	result.value = value;
	result.time_searches = (end-start).count() / 1000000000.0;
	result.number_searches = number_searches;
	return result;
}

auto distance(const Curve &curve1, const Curve &curve2, distance_t ub, distance_t lb, const distance_t eps, bool benchmark) -> Result {
	Result result;
	auto start = boost::chrono::process_real_cpu_clock::now();
	
	distance_t split = (ub + lb)/2;
	std::size_t number_searches = 0;
	
	if (benchmark) {

		if (ub - lb > eps) {
			auto infty = std::numeric_limits<distance_t>::infinity();
			std::vector<std::vector<distance_t>> reachable1(curve1.size()-1, std::vector<distance_t>(curve2.size(), infty));
			std::vector<std::vector<distance_t>> reachable2(curve1.size(), std::vector<distance_t>(curve2.size()-1, infty));

			//Binary search over the feasible distances
			while (ub - lb > eps) {
				++number_searches;
				split = (ub + lb)/2;
				auto isLessThan = false;
				{
					isLessThan = _lessThan(split, curve1, curve2, reachable1, reachable2);
				}
				if (isLessThan) {
					ub = split;
				}
				else {
					lb = split;
				}
				#if DEBUG
				std::cout << "narrowed to [" << lb << ", " << ub << "]" << std::endl;
				#endif
			}
		}
	} else {
		if (ub - lb > eps) {
			auto infty = std::numeric_limits<distance_t>::infinity();
			std::vector<std::vector<distance_t>> reachable1(curve1.size()-1, std::vector<distance_t>(curve2.size(), infty));
			std::vector<std::vector<distance_t>> reachable2(curve1.size(), std::vector<distance_t>(curve2.size()-1, infty));

			//Binary search over the feasible distances
			while (ub - lb > eps) {
				++number_searches;
				split = (ub + lb)/2;
				auto isLessThan = _lessThan(split, curve1, curve2, reachable1, reachable2);
				if (isLessThan) {
					ub = split;
				}
				else {
					lb = split;
				}
				#if DEBUG
				std::cout << "narrowed to [" << lb << ", " << ub << "]" << std::endl;
				#endif
			}
		}
	
	}
	auto value = std::round((ub + lb)/2.*1000)/1000;
	auto end = boost::chrono::process_real_cpu_clock::now();
	result.value = value;
	result.time_searches = (end-start).count() / 1000000000.0;
	result.number_searches = number_searches;
	return result;
}

bool _lessThan_cuda(const distance_t distance, Curve const& curve1, 
		Curve const& curve2, std::vector<std::vector<distance_t>> &reachable1, 
		std::vector<std::vector<distance_t>> &reachable2, Cuda_intersection &cuda) {
	assert(curve1.size() >= 2);
	assert(curve2.size() >= 2);
	
	distance_t dist_sqr = distance * distance;
	auto infty = std::numeric_limits<distance_t>::infinity();

	if (curve1[0].dist_sqr(curve2[0]) > dist_sqr or curve1.back().dist_sqr(curve2.back()) > dist_sqr) return false;

	for (auto &elem: reachable1) {
		for (std::size_t i = 0; i < elem.size(); ++i) {
			elem[i] = infty;
		}
	}
	
	for (auto &elem: reachable2) {
		for (std::size_t i = 0; i < elem.size(); ++i) {
			elem[i] = infty;
		}
	}
	
	for (size_t i = 0; i < curve1.size() - 1; ++i) {
		reachable1[i][0] = 0.;
		if (curve2[0].dist_sqr(curve1[i+1]) > dist_sqr) { break; }
	}
	for (size_t j = 0; j < curve2.size() - 1; ++j) {
		reachable2[0][j] = 0.;
		if (curve1[0].dist_sqr(curve2[j+1]) > dist_sqr) { break; }
	}



	//Some index_combinations curve1_index curve2_index have 2 results. Thats the case if the following 2 if statements are both true:
	//if (i < curve1.size() - 1 && j > 0) and if (j < curve2.size() - 1 and i > 0)
	//If it happens to be the case that one index combination has 2 results then the first result is simply stored at:
	//results[curve2_index * curve1.size() + curve1_index]. But if the second result is stored at the following index:
	//results[(curve2_index * curve1.size() + curve1_index) + curve1.size()*curve2.size()]
	cuda.intersection_interval_cuda(distance);
	distance_t *results_p = cuda.get_host_results_p();

	//Interval free_int;
	distance_t free_int_cuda_begin;
	distance_t free_int_cuda_end;
	for (size_t i = 0; i < curve1.size(); ++i) {
		for (size_t j = 0; j < curve2.size(); ++j) {
			bool case1_true = false;
			if (i < curve1.size() - 1 && j > 0) {
				case1_true = true;
				//Most time consuming part by far!
				//We could start calculating this all at once. If we have p cuda prozessors and have to calculate this n times
				//then it takes us n/p  time. We speed up our algorithm by p!!!!
				free_int_cuda_begin = results_p[(2*(j*curve1.size() + i))];
				free_int_cuda_end = results_p[(2*(j*curve1.size() + i))+1];

				if (not (free_int_cuda_begin > free_int_cuda_end)) {
					if (reachable2[i][j-1] != infty) {
						reachable1[i][j] = free_int_cuda_begin;
					}
					else if (reachable1[i][j-1] <= free_int_cuda_end) {
						reachable1[i][j] = std::max(free_int_cuda_begin, reachable1[i][j-1]);
					}
				}
			}
			if (j < curve2.size() - 1 and i > 0) {
				if(case1_true){
					free_int_cuda_begin = results_p[(2*(j*curve1.size() + i + curve1.size() *curve2.size()))];
					free_int_cuda_end = results_p[(2*(j*curve1.size() + i + curve1.size() *curve2.size()))+1];
				}else{
					free_int_cuda_begin = results_p[(2*(j*curve1.size() + i))];
					free_int_cuda_end = results_p[(2*(j*curve1.size() + i))+1];
				}
					
				if (not (free_int_cuda_begin > free_int_cuda_end)) {
					if (reachable1[i-1][j] != infty) {
						reachable2[i][j] = free_int_cuda_begin;
					}
					else if (reachable2[i-1][j] <= free_int_cuda_end) {
						reachable2[i][j] = std::max(free_int_cuda_begin, reachable2[i-1][j]);
					}
				}
			}
		}
	}

	assert((reachable1.back().back() < infty) == (reachable2.back().back() < infty));

	return reachable1.back().back() < infty;
}

bool _lessThan(const distance_t distance, Curve const& curve1, Curve const& curve2, 
		std::vector<std::vector<distance_t>> &reachable1, std::vector<std::vector<distance_t>> &reachable2) {
	assert(curve1.size() >= 2);
	assert(curve2.size() >= 2);
	
	distance_t dist_sqr = distance * distance;
	auto infty = std::numeric_limits<distance_t>::infinity();

	if (curve1[0].dist_sqr(curve2[0]) > dist_sqr or curve1.back().dist_sqr(curve2.back()) > dist_sqr) return false;

	for (auto &elem: reachable1) {
		for (std::size_t i = 0; i < elem.size(); ++i) {
			elem[i] = infty;
		}
	}
	
	for (auto &elem: reachable2) {
		for (std::size_t i = 0; i < elem.size(); ++i) {
			elem[i] = infty;
		}
	}
	
	for (size_t i = 0; i < curve1.size() - 1; ++i) {
		reachable1[i][0] = 0.;
		if (curve2[0].dist_sqr(curve1[i+1]) > dist_sqr) { break; }
	}
	for (size_t j = 0; j < curve2.size() - 1; ++j) {
		reachable2[0][j] = 0.;
		if (curve1[0].dist_sqr(curve2[j+1]) > dist_sqr) { break; }
	}

	Interval free_int;
	for (size_t i = 0; i < curve1.size(); ++i) {
		for (size_t j = 0; j < curve2.size(); ++j) {
			if (i < curve1.size() - 1 and j > 0) {
				{
					free_int = IntersectionAlgorithm::intersection_interval(curve2[j], distance, curve1[i], curve1[i+1]);
				}
				if (not free_int.is_empty()) {
					if (reachable2[i][j-1] != infty) {
						reachable1[i][j] = free_int.begin;
					}
					else if (reachable1[i][j-1] <= free_int.end) {
						reachable1[i][j] = std::max(free_int.begin, reachable1[i][j-1]);
					}
				}
			}
			if (j < curve2.size() - 1 and i > 0) {
				{
					free_int = IntersectionAlgorithm::intersection_interval(curve1[i], distance, curve2[j], curve2[j+1]);
				}
				if (not free_int.is_empty()) {
					if (reachable1[i-1][j] != infty) {
						reachable2[i][j] = free_int.begin;
					}
					else if (reachable2[i-1][j] <= free_int.end) {
						reachable2[i][j] = std::max(free_int.begin, reachable2[i-1][j]);
					}
				}
			}
		}
	}

	assert((reachable1.back().back() < infty) == (reachable2.back().back() < infty));

	return reachable1.back().back() < infty;
}


}

namespace Discrete {
	
	auto _dp(std::vector<std::vector<distance_t>> &a, const std::size_t i, const std::size_t j, const Curve &curve1, const Curve &curve2) -> distance_t {
		if (a[i][j] > -1) return a[i][j];
		else if (i == 0 and j == 0) return curve1[i].dist_sqr(curve2[j]);
		else if (i > 0 and j == 0) return std::max(_dp(a, i-1, 0, curve1, curve2), curve1[i].dist_sqr(curve2[j]));
		else if (i == 0 and j > 0) return std::max(_dp(a, 0, j-1, curve1, curve2), curve1[i].dist_sqr(curve2[j]));
		else {
			a[i][j] = std::max(
						std::min(
							std::min(_dp(a, i-1, j, curve1, curve2), 
								_dp(a, i-1, j-1, curve1, curve2)), 
							_dp(a, i, j-1, curve1, curve2)), 
						curve1[i].dist_sqr(curve2[j]));
		}
		return a[i][j];
	}
	
	auto distance(const Curve &curve1, const Curve &curve2) -> Result {
		Result result;
		auto start = boost::chrono::process_real_cpu_clock::now();
		std::vector<std::vector<distance_t>> a(curve1.size(), std::vector<distance_t>(curve2.size(), -1));
		auto value = std::sqrt(_dp(a, curve1.size()-1, curve2.size()-1, curve1, curve2));
		auto end = boost::chrono::process_real_cpu_clock::now();
		result.time = (end-start).count() / 1000000000.0;
		result.value = value;
		return result;
		
	}
}

}
