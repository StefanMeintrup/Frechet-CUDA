#pragma once

#include "curve.hpp"
#include "random.hpp"

namespace JLTransform {
	
	Curves transform_naive(const Curves &in, const double epsilon, const bool empirical_k = true) {
		
		if (in.empty()) return in;
		
		auto rg = Gauss_Random_Generator<coordinate_t>(0, 1);
		
		auto number_points = 0;
		for (auto &elem: in) number_points += elem.size();
		
		const auto epsilonsq = epsilon * epsilon;		
        const auto epsiloncu = epsilonsq * epsilon;
		const auto new_number_dimensions = empirical_k ? std::ceil(2 * std::log(number_points) * 1/epsilonsq):
            std::ceil(4 * std::log(number_points) * 1 /((epsilonsq/2) - (epsiloncu/3)));
													
		std::vector<std::vector<coordinate_t>> mat (new_number_dimensions);
		
		#if DEBUG
		std::cout << "populating " << new_number_dimensions << "x" << in[0].dimensions() << " matrix" << std::endl;
		#endif
		
		for (auto &elem: mat) elem = rg.get(in[0].dimensions());
					
		Curves result;
		
		auto sqrtk = std::sqrt(new_number_dimensions);
		
		for (std::size_t l = 0; l < in.size(); ++l) {
			result.push_back(Curve(new_number_dimensions));
			
			for (std::size_t i = 0; i < in[l].size(); ++i) {
				std::vector<coordinate_t> new_coords(new_number_dimensions);
				
				for (std::size_t j = 0; j < new_number_dimensions; ++j) {
					new_coords[j] = mat[j][0] * in[l][i][0];
					
					for (std::size_t k = 1; k < in[l].dimensions(); ++k) {
						new_coords[j] += mat[j][k] * in[l][i][k];
					}
					
					new_coords[j] /= sqrtk;
					
				}
				
				Point new_point(new_coords);
				result[l].push_back(new_point);
			}
			
			#if DEBUG
			std::cout << "projected curve no. " << l << " from " << in[l].dimensions() << " to " << new_number_dimensions << " dimensions" << std::endl;
			#endif
			
		}
		
		return result;
	}
	
}
