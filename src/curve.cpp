#include "curve.hpp"

Curve::Curve(const Points& points, dimensions_t dimensions) : 
	points(points), prefix_length(points.size()), number_dimensions(dimensions) {
	if (points.empty()) { 
		std::cerr << "warning: constructed empty curve" << std::endl;
		return; 
	}

	prefix_length[0] = 0;

	for (std::size_t i = 1; i < points.size(); ++i)
	{
		auto segment_distance = points[i - 1].dist(points[i]);
		prefix_length[i] = prefix_length[i - 1] + segment_distance;
	}
	#if DEBUG
	std::cout << "constructed curve of complexity " << points.size() << std::endl;
	#endif
}

Curve::Curve(const np::ndarray &in): number_dimensions(in.shape(1)) {
            auto dimensions = in.get_nd();
            if (dimensions != 2 or in.get_dtype() != np::dtype::get_builtin<coordinate_t>()){
                std::cerr << "A Polygonal_Curve requires an 2-dimensional numpy array of type double."<< std::endl;
                std::cerr << "Current dimensiont: " << dimensions << std::endl;
                return;
            }
			auto number_points = in.shape(0);
			auto point_size = in.shape(1);
			
			#if DEBUG
			std::cout << "constructing curve of size " << number_points << " and " << point_size << " dimensions" << std::endl;
			#endif
			
            auto strides0 = in.strides(0) / sizeof(coordinate_t);
            auto strides1 = in.strides(1) / sizeof(coordinate_t);
            
            auto data = reinterpret_cast<const coordinate_t*>(in.get_data());
			
            points = Points(number_points);

            //Fill the points
            for (index_type i = 0; i < number_points; ++i, data += strides0) {
                points[i] = Point(std::vector<coordinate_t> (point_size));
                
                auto coord_data = data;
                
                for(index_type j = 0; j < point_size; ++j, coord_data += strides1){
                  points[i].getCoordinates()[j] = *coord_data;
                }
            }
            
			prefix_length =  std::vector<distance_t> (points.size());
			for (std::size_t i = 1; i < points.size(); ++i)
			{
				auto segment_distance = points[i - 1].dist(points[i]);
				prefix_length[i] = prefix_length[i - 1] + segment_distance;
			}
			if (points.empty()) { 
				std::cerr << "warning: constructed empty curve" << std::endl;
			return; 
	}
}

void Curve::push_back(Point const& point)
{
	if (prefix_length.size()) {
		auto segment_distance = points.back().dist(point);
		prefix_length.push_back(prefix_length.back() + segment_distance);
	}
	else {
		prefix_length.push_back(0);
	}
	
	points.push_back(point);
}

std::ostream& operator<<(std::ostream& out, const Curve& curve)
{
    out << "[";
	for (auto const& point: curve) {
		out << point << ", ";
	}
    out << "]";

    return out;
}

