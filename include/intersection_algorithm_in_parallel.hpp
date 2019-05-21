#pragma once

#include "geometry_basics.hpp"
#include "curve.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Cuda_intersection{
    const bool is_last = true;
    const bool is_not_last = false;
    public:
        Cuda_intersection(const Curve& curve1, const Curve& curve2, distance_t *host_res_p);

        void intersection_interval_cuda(
            distance_t radius
        ); 
        cudaError_t intersection_interval_call_gpu( 
            distance_t radius
        );

        distance_t* get_host_results_p(){
            return host_results_p;
        };

        void free_memory(); 
    
    private:

        std::vector<distance_t *> dev_radius_p;
        std::vector<coordinate_t *> dev_points_curve1_p;
        std::vector<coordinate_t *> dev_points_curve2_p;
        std::vector<curve_size_t *> dev_curve1_size_p;
        std::vector<curve_size_t *> dev_curve2_size_p;
        std::vector<curve_size_t *> dev_point_dimensions_p;
        std::vector<bool *> dev_is_last_kernel_p;
        std::vector<distance_t *> dev_results_p;

        std::vector<curve_size_t> curve1_size;
        std::vector<curve_size_t> curve2_size;
        std::vector<curve_size_t> curve1_start_index;
        std::vector<curve_size_t> curve1_end_index;
        std::vector<curve_size_t> curve2_start_index;
        std::vector<curve_size_t> curve2_end_index;

        curve_size_t point_dimensions;
        distance_t *host_results_p;
        int number_devices;

        coordinate_t *points_curve1_p;
        coordinate_t *points_curve2_p;

        bool is_buffers_free = false;

};

int calculate_optimal_blocksize(Cuda_intersection cuda_int, distance_t distance);
