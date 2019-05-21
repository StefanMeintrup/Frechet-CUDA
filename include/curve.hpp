#pragma once

#include <iostream> 

#include <boost/python/numpy.hpp>
#include <boost/python.hpp>

#include "geometry_basics.hpp"

namespace np = boost::python::numpy;
namespace p = boost::python;

// Represents a trajectory. Additionally to the points given in the input file,
// we also store the length of any prefix of the trajectory.
class Curve {
public:
	typedef unsigned long index_type;

	Curve(dimensions_t dimensions) : number_dimensions{dimensions} {}
    Curve(const Points& points, dimensions_t dimensions);
	Curve(const np::ndarray &in);

    inline std::size_t size() const { return points.size(); }
	inline bool empty() const { return points.empty(); }
	inline std::size_t dimensions() const { return number_dimensions; }
    inline Point const& operator[](const std::size_t i) const { return points[i]; }

    inline distance_t curve_length(const std::size_t i, const std::size_t j) const
		{ return prefix_length[j] - prefix_length[i]; }

    inline Point front() const { return points.front(); }
    inline Point back() const { return points.back(); }

    void push_back(const Point &point);

	inline Points::iterator begin() { return points.begin(); }
	inline Points::iterator end() { return points.end(); }
	inline Points::const_iterator begin() const { return points.cbegin(); }
	inline Points::const_iterator end() const { return points.cend(); }
	
private:
	const dimensions_t number_dimensions;
	Points points;
    std::vector<distance_t> prefix_length;
};

class Curves : public std::vector<Curve> {
public:
	inline Curve get(std::size_t i) const {
		return this->operator[](i);
	}
};

std::ostream& operator<<(std::ostream& out, const Curve& curve);
