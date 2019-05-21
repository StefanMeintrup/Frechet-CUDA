#include "geometry_basics.hpp"

//
// Point
//

std::ostream& operator<<(std::ostream& out, const Point &p)
{
    out << std::setprecision (15)
		<< "(";
	for( unsigned long i = 0; i < p.getCoordinates().size(); i++){
		out << std:: setprecision (15) << p.getCoordinates().at(i);
		if( i != p.getCoordinates().size() - 1){
			out << ",";
		}
	}
	out << ")";

    return out;
}

//
// Interval
//

std::ostream& operator<<(std::ostream& out, const Interval& interval)
{
    out << std::setprecision (15)
		<< "(" << interval.begin << ", " << interval.end << ")";

    return out;
}
