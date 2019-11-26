#include <boost/python.hpp>

#include "frechet.hpp"
#include "curve.hpp"
#include "jl_transform.hpp"
#include "clustering.hpp"

using namespace boost::python;
namespace np = boost::python::numpy;
namespace fc = Frechet::Continuous;
namespace fd = Frechet::Discrete;

const distance_t eps = 0.001;

fc::Result compute_distance(const Curve &curve1, const Curve &curve2) {	
	
	//const distance_t eps = 0.001;
	distance_t lb;
	fd::Result ub;

    lb = std::sqrt(std::max(curve1.front().dist_sqr(curve2.front()), curve1.back().dist_sqr(curve2.back())));
    ub = fd::distance(curve1, curve2);

	#if DEBUG
	std::cout << "narrowed to [" << lb << ", " << ub.value << "]" << std::endl;
	#endif
	
	auto dist = fc::distance(curve1, curve2, ub.value, lb, eps);
	dist.time_bounds = ub.time;

	return dist;
}

fc::Result compute_distance_parallel(const Curve &curve1, const Curve &curve2) {	
	
	//const distance_t eps = 0.001;
	distance_t lb;
	fd::Result ub;

	lb = std::sqrt(std::max(curve1.front().dist_sqr(curve2.front()), curve1.back().dist_sqr(curve2.back())));
	ub = fd::distance(curve1, curve2);

	#if DEBUG
	std::cout << "narrowed to [" << lb << ", " << ub.value << "]" << std::endl;
	#endif
	
	auto dist = fc::distance_cuda(curve1, curve2, ub.value, lb, eps);
	dist.time_bounds = ub.time;

	return dist;
}

Curves jl_transform(const Curves &in, const double epsilon) {
	
	Curves curvesrp = JLTransform::transform_naive(in, epsilon);
	
	return curvesrp;
}

Clustering::Clustering_Result kcenter(const std::size_t num_centers, const Curves &in) {
	auto result = Clustering::gonzalez(num_centers, in);
	
	return result;
}

Clustering::Clustering_Result onemedian_approx(const double epsilon, const Curves &in) {
	
	auto result = Clustering::one_median_approx(epsilon, in);
	
	return result;
}

Clustering::Clustering_Result onemedian_exhaust(const Curves &in) {

	auto result = Clustering::one_median_exhaustive(in);
    
	return result;
}

Clustering::Clustering_Result kmedian(const std::size_t num_centers, const Curves &in) {

	auto result = Clustering::arya(num_centers, in);
	
	return result;
}

Clustering::Clustering_Result kcenter_with_assignment(const std::size_t num_centers, const Curves &in) {
	
	auto result = Clustering::gonzalez(num_centers, in, false, true);
	
	return result;
}

BOOST_PYTHON_MODULE(frechet_cuda)
{
    Py_Initialize();
    np::initialize();
    
    scope().attr("epsilon") = eps;

	class_<Curves>("Curves", init<>())
		.def("add", static_cast<void (Curves::*)(const Curve&)>(&Curves::push_back))
		.def("__getitem__", &Curves::get)
        .def("__len__", &Curves::size)
	;
	
	class_<Curve>("Curve", init<np::ndarray>())
		.add_property("dimensions", &Curve::dimensions)
		.add_property("points", &Curve::size)
        .def("__len__", &Curve::size)
	;
	
	class_<Clustering::Clustering_Result>("Clustering_Result", init<>())
		.def("__getitem__", &Clustering::Clustering_Result::get)
		.def("__len__", &Clustering::Clustering_Result::size)
		.add_property("value", &Clustering::Clustering_Result::value)
		.add_property("running_time", &Clustering::Clustering_Result::running_time)
		.add_property("assignment", &Clustering::Clustering_Result::assignment)
	;
	
	class_<Frechet::Continuous::Result>("Frechet_Result", init<>())
		.add_property("time_searches", &Frechet::Continuous::Result::time_searches)
		.add_property("time_bounds", &Frechet::Continuous::Result::time_bounds)
		.add_property("number_searches", &Frechet::Continuous::Result::number_searches)
		.add_property("value", &Frechet::Continuous::Result::value)
	;
	
	class_<Clustering::Cluster_Assignment>("Clustering_Assignment", init<>())
		.def("__len__", &Clustering::Cluster_Assignment::size)
		.def("count", &Clustering::Cluster_Assignment::count)
		.def("__getitem__", &Clustering::Cluster_Assignment::get)
    ;
    
    def("jl_transform", jl_transform);
    def("compute_distance", compute_distance_parallel);
    def("compute_distance_sequential", compute_distance);
    def("kcenter", kcenter);
    def("kcenter_with_assignment", kcenter_with_assignment);
    def("kmedian", kmedian);
    def("onemedian_approx", onemedian_approx);
    def("onemedian_exhaustive", onemedian_exhaust);
}
