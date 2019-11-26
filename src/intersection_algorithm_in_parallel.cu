#include <math.h>

#include "intersection_algorithm_in_parallel.hpp"

#define BLOCKSIZE 64

__global__ void cuda_kernel1_calculate_intervals(
    distance_t *results_p,
    coordinate_t *points_curve1_p,
    coordinate_t *points_curve2_p,
    curve_size_t *curve1_size_p,
    curve_size_t *curve2_size_p,
    curve_size_t *point_dimensions_p,
    distance_t *radius_p,
    bool *is_last
);


Cuda_intersection::Cuda_intersection(const Curve& curve1,const Curve& curve2, distance_t *host_res_p, distance_t eps) : eps{eps} {

        cudaError_t cudaStatus;
        host_results_p = host_res_p;
        point_dimensions = curve1[0].coordinates.size();
        points_curve1_p = (coordinate_t*)malloc(sizeof(coordinate_t)*point_dimensions*curve1.size());
        points_curve2_p = (coordinate_t*)malloc(sizeof(coordinate_t)*point_dimensions*curve2.size());

        //If we have the opportunity to use multiple gpus then we will split curve2 onto the gpus, but give each gpu
        //the complete first curve.
        cudaStatus = cudaGetDeviceCount(&number_devices);
        if(cudaStatus != cudaSuccess){
            std::cerr << "cudaGetDeviceCout failed! Do you have CUDA-capable GPU installed?" << std::endl;
            goto Error;
        }
        //If splitting curve2 ist not reasonable, do not do it
        if(curve2.size() <= 2*number_devices){
            number_devices = 1;
        }

        dev_radius_p.reserve(number_devices);
        dev_points_curve1_p.reserve(number_devices);
        dev_points_curve2_p.reserve(number_devices);
        dev_curve1_size_p.reserve(number_devices);
        dev_curve2_size_p.reserve(number_devices);
        dev_point_dimensions_p.reserve(number_devices);
        dev_results_p.reserve(number_devices);
        dev_is_last_kernel_p.reserve(number_devices);

        curve1_size.reserve(number_devices);
        curve2_size.reserve(number_devices);
        curve1_start_index.reserve(number_devices);
        curve2_start_index.reserve(number_devices);
        curve1_end_index.reserve(number_devices);
        curve2_end_index.reserve(number_devices);

        curve1_start_index[0] = 0;
        curve2_start_index[0] = 0;
        if(number_devices != 1){
            curve1_end_index[0] = curve1.size()-1;
            if(ceil(curve2.size() / number_devices) > curve2.size()-1){
                curve2_end_index[0] = curve2.size() -1;
            }else{
                curve2_end_index[0] = ceil(curve2.size() / number_devices);
            }
        }else{
            curve1_end_index[0] = curve1.size() - 1;
            curve2_end_index[0] = curve2.size() - 1;
        }
        
        curve1_size[0] = curve1_end_index[0]-curve1_start_index[0]+1;
        curve2_size[0] = curve2_end_index[0]-curve2_start_index[0]+1;

        //Fill the points:
        for( curve_size_t i = 0; i < curve1.size(); i++){
            for( curve_size_t j = 0; j < point_dimensions; j++){
                points_curve1_p[i*point_dimensions + j] = curve1[i].coordinates[j];
            }
        }
    
        for( curve_size_t i = 0; i < curve2.size(); i++){
            for( curve_size_t j = 0; j < point_dimensions; j++){
                points_curve2_p[i*point_dimensions + j] = curve2[i].coordinates[j];
            }
        }

        for(short device_nbr = 1; device_nbr < number_devices; device_nbr++){
            curve1_start_index[device_nbr] = 0;
            curve2_start_index[device_nbr] = curve2_end_index[device_nbr-1] - 1;
            curve1_end_index[device_nbr] = curve1.size() - 1;
            if(not device_nbr == number_devices - 1){
                curve2_end_index[device_nbr] = curve2_start_index[device_nbr-1] + ceil(curve2.size() / number_devices);
            }else{
                curve2_end_index[device_nbr] = curve2.size() - 1;
            }
            curve1_size[device_nbr] = curve1_end_index[device_nbr]-curve1_start_index[device_nbr]+1;
            curve2_size[device_nbr] = curve2_end_index[device_nbr]-curve2_start_index[device_nbr]+1;
        }
        #if DEBUG
        std::cout << "NUMBER DEVICES: " << number_devices << std::endl;
        #endif

        for(short device_nbr = 0; device_nbr < number_devices; device_nbr++){
            //Choose which GPU to run on, change this on a multi-GPU system if needed!
            cudaStatus = cudaSetDevice(device_nbr);
            if(cudaStatus != cudaSuccess){
                std::cerr << "CUDASetDevice failed! Do you have CUDA-capable GPU installed?" << std::endl;
            }

            //Allocate buffers on the GPU
            cudaStatus = cudaMalloc((void**)&dev_results_p[device_nbr], sizeof(distance_t)*2*(2 * curve1_size[device_nbr] * curve2_size[device_nbr]));
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMalloc dev_results failed!" << std::endl;
                goto Error;
            }

            cudaStatus = cudaMalloc((void**)&dev_points_curve1_p[device_nbr], sizeof(coordinate_t)* point_dimensions * curve1_size[device_nbr]);
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMalloc dev_points_curve1 failed!" << std::endl;
                goto Error;
            }
        
            cudaStatus = cudaMalloc((void**)&dev_points_curve2_p[device_nbr], sizeof(coordinate_t)* point_dimensions * curve2_size[device_nbr]);
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMalloc dev_points_curve2 failed!" << std::endl;
                goto Error;
            }
        
            cudaStatus = cudaMalloc((void**)&dev_curve1_size_p[device_nbr], sizeof(curve_size_t));
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMalloc dev_curve1_size_p failed!" << std::endl;
                goto Error;
            }
        
            cudaStatus = cudaMalloc((void**)&dev_curve2_size_p[device_nbr], sizeof(curve_size_t));
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMalloc dev_curve2_size_p failed!" << std::endl;
                goto Error;
            }
    
            cudaStatus = cudaMalloc((void**)&dev_point_dimensions_p[device_nbr], sizeof(curve_size_t));
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMalloc dev_point_dimensions_p failed!" << std::endl;
                goto Error;
            }
            
            cudaStatus = cudaMalloc((void**)&dev_radius_p[device_nbr], sizeof(distance_t));
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMalloc dev_radius failed!" << std::endl;
                goto Error;
            }

            cudaStatus = cudaMalloc((void**)&dev_is_last_kernel_p[device_nbr], sizeof(bool));
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMalloc dev_is_last_kernel_p failed!" << std::endl;
                goto Error;
            }

            //Copy Data into device memory
            cudaStatus = cudaMemcpy(dev_points_curve1_p[device_nbr], &points_curve1_p[curve1_start_index[device_nbr]], sizeof(coordinate_t)* point_dimensions * (curve1_end_index[device_nbr]-curve1_start_index[device_nbr]+1), cudaMemcpyHostToDevice );
            if (cudaStatus != cudaSuccess ){
                std::cerr << "CudaMemcpy curve1_points to dev_points_curve1_p failed!" << std::endl;
                goto Error;
            }
        
            cudaStatus = cudaMemcpy(dev_points_curve2_p[device_nbr], &points_curve2_p[curve2_start_index[device_nbr]], sizeof(coordinate_t)* point_dimensions * (curve2_end_index[device_nbr]-curve2_start_index[device_nbr]+1), cudaMemcpyHostToDevice );
            if (cudaStatus != cudaSuccess ){
                std::cerr << "CudaMemcpy curve2_points to dev_points_curve2_p failed!" << std::endl;
                goto Error;
            }
        
            cudaStatus = cudaMemcpy(dev_curve1_size_p[device_nbr], &curve1_size[device_nbr], sizeof(curve_size_t), cudaMemcpyHostToDevice );
            if (cudaStatus != cudaSuccess ){
                std::cerr << "CudaMemcpy curve.size() host to device failed!" << std::endl;
                goto Error;
            }
            cudaStatus = cudaMemcpy(dev_curve2_size_p[device_nbr], &curve2_size[device_nbr], sizeof(curve_size_t), cudaMemcpyHostToDevice );
            if (cudaStatus != cudaSuccess ){
                std::cerr << "CudaMemcpy curve2.size() host to device failed!" << std::endl;
                goto Error;
            }
            cudaStatus = cudaMemcpy(dev_point_dimensions_p[device_nbr], &point_dimensions, sizeof(curve_size_t), cudaMemcpyHostToDevice );
            if (cudaStatus != cudaSuccess ){
                std::cerr << "CudaMemcpy point_dimensions host to device failed!" << std::endl;
                goto Error;
            }

            if(device_nbr == number_devices-1){
                cudaStatus = cudaMemcpy(dev_is_last_kernel_p[device_nbr], &is_last ,sizeof(bool), cudaMemcpyHostToDevice);
                if(cudaStatus != cudaSuccess){
                    std::cerr << "CudaMemcpy dev_is_last_kernel_p failed!" << std::endl;
                    goto Error;
                }
            }else{
                cudaStatus = cudaMemcpy(dev_is_last_kernel_p[device_nbr], &is_not_last ,sizeof(bool), cudaMemcpyHostToDevice);
                if(cudaStatus != cudaSuccess){
                    std::cerr << "CudaMemcpy dev_is_last_kernel_p failed!" << std::endl;
                    goto Error;
                }
            }
        }
        goto NoError;

        Error:
            free_memory();
        NoError:
			return;
}

void Cuda_intersection::free_memory(){
    if(not is_buffers_free){
        for(short device_nbr = 0; device_nbr < number_devices; device_nbr++){
            cudaFree(dev_results_p[device_nbr]);
            cudaFree(dev_points_curve1_p[device_nbr]);
            cudaFree(dev_points_curve2_p[device_nbr]);
            cudaFree(dev_radius_p[device_nbr]);
            cudaFree(dev_curve1_size_p[device_nbr]);
            cudaFree(dev_curve2_size_p[device_nbr]);
            cudaFree(dev_point_dimensions_p[device_nbr]);
            cudaFree(dev_is_last_kernel_p[device_nbr]);
        }
        free(points_curve1_p);
        free(points_curve2_p);
        is_buffers_free = true;
    }
}

void Cuda_intersection::intersection_interval_cuda(
    distance_t radius
){
    cudaError_t cudaStatus = intersection_interval_call_gpu(radius);
    if(cudaStatus != cudaSuccess){
        std::cerr << "intersection_interval_call_gpu failed!" << std::endl;
    }
}

__global__ void cuda_kernel1_calculate_intervals(
    distance_t *results_p,
    coordinate_t *points_curve1_p,
    coordinate_t *points_curve2_p,
    curve_size_t *curve1_size_p,
    curve_size_t *curve2_size_p,
    curve_size_t *point_dimensions_p,
    distance_t *radius_p,
    bool *is_last,
    distance_t eps
){

  thread_id_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  curve_size_t curve1_index = (curve_size_t)(thread_id % (*curve1_size_p));
  curve_size_t curve2_index = (curve_size_t)(thread_id / (*curve1_size_p));


    if(curve1_index >= *curve1_size_p or curve2_index >= *curve2_size_p){
        return; 
    }
    if(curve1_index == *curve1_size_p - 1 and not *is_last){
        return;
    }
    if(curve2_index == *curve2_size_p - 1 and not *is_last){
        return;
    }

  coordinate_t *circle_center_p;
  coordinate_t *line_start_p;
  coordinate_t *line_end_p;

  //If this is not getting set then the interval does not have to be calculated
  bool need_to_calculate_v1 = false;
  bool need_to_calculate_v2 = false;
  short reps = 0;

  if(curve1_index < *curve1_size_p -1 and curve2_index > 0){
    need_to_calculate_v1 = true;
  }

  if(curve2_index < *curve2_size_p -1 and curve1_index > 0){
    need_to_calculate_v2 = true;
  }

  if(need_to_calculate_v1 and need_to_calculate_v2){
      reps = 2;
  }else if(need_to_calculate_v1){
      reps = 1;
  }else if(need_to_calculate_v2){
      reps = 1;
  }
  //Do the intersection algorithm if needed
  short current_repetition = 1;
  while(current_repetition <= reps){
        if(need_to_calculate_v1){
            need_to_calculate_v1 = false;
            circle_center_p = &points_curve2_p[curve2_index * *point_dimensions_p];
            line_start_p = &points_curve1_p[curve1_index * *point_dimensions_p];
            line_end_p = &points_curve1_p[(curve1_index+1) * *point_dimensions_p];
        }
        else if(need_to_calculate_v2){
            need_to_calculate_v2 = false;
            circle_center_p = &points_curve1_p[curve1_index * *point_dimensions_p];
            line_start_p = &points_curve2_p[curve2_index * *point_dimensions_p];
            line_end_p = &points_curve2_p[(curve2_index+1) * *point_dimensions_p];
        }else {
            return;
        }


        //const distance_t eps = 0.001 / 4;
        const distance_t save_eps = 0.5 * eps;
        const distance_t save_eps_half = 0.25 * eps;

        distance_t radius_sqr = *radius_p * *radius_p;
        distance_t dist_a = 0;
        distance_t dist_b = 0;
        distance_t dist_c = - radius_sqr;
        distance_t mid;
        distance_t discriminant;
        distance_t sqrt_discr = 0.;
        distance_t begin, end;
        bool smallDistAtZero;
        bool smallDistAtOne;
        bool smallDistAtMid;
        bool sqrt_discr_computed = false;
        

        for(curve_size_t coordinate = 0; coordinate < *point_dimensions_p; coordinate++){
            coordinate_t end_minus_start_coordinate = line_end_p[coordinate] - line_start_p[coordinate];
            dist_a += end_minus_start_coordinate * end_minus_start_coordinate;
            dist_b += (line_start_p[coordinate] - circle_center_p[coordinate]) * end_minus_start_coordinate;
            dist_c += powf(line_start_p[coordinate] - circle_center_p[coordinate], 2);
        }

        mid = - dist_b / dist_a;
        discriminant = mid * mid - dist_c / dist_a;

        distance_t circle_center_dist0_sqr = 0;
        distance_t circle_center_dist1_sqr = 0;
        distance_t circle_center_dist_mid_sqr = 0;

        for(curve_size_t coordinate = 0; coordinate < *point_dimensions_p; coordinate++){
            distance_t coordinate_for_dist0 = line_start_p[coordinate] * 1.;
            distance_t coordinate_for_dist1 = line_end_p[coordinate] * 1.;
            distance_t coordinate_for_dist_mid = line_start_p[coordinate] * (1. - mid) + line_end_p[coordinate] * mid;
            circle_center_dist0_sqr += powf(circle_center_p[coordinate] - coordinate_for_dist0, 2);
            circle_center_dist1_sqr += powf(circle_center_p[coordinate] - coordinate_for_dist1, 2);
            circle_center_dist_mid_sqr += powf(circle_center_p[coordinate] - coordinate_for_dist_mid, 2);
            if(coordinate == *point_dimensions_p - 1){
                smallDistAtZero = circle_center_dist0_sqr <= radius_sqr;
                smallDistAtOne = circle_center_dist1_sqr <= radius_sqr;
                smallDistAtMid = circle_center_dist_mid_sqr <= radius_sqr;
            }
        }

        if(smallDistAtZero and smallDistAtOne){
            if(current_repetition == 2){
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index + (*curve1_size_p)*(*curve2_size_p))] = 0;
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index + (*curve1_size_p)*(*curve2_size_p))+1] = 1;
                return;
            }else{
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index )] = 0;
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index)+1] = 1;
                current_repetition++;
                continue;
            }    
        }

        if(not smallDistAtMid and smallDistAtZero){
            mid = 0.;
            smallDistAtMid = true;
        }else if(not smallDistAtMid and smallDistAtOne){
            mid = 1.;
            smallDistAtMid = true;
        }

        if(not smallDistAtMid){
            if(current_repetition == 2){
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index + (*curve1_size_p)*(*curve2_size_p))] = 1.;
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index + (*curve1_size_p)*(*curve2_size_p))+1] = 0.;
                return;
            }else{
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index )] = 1.;
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index )+1] = 0.;  
                current_repetition++;     
                continue;
            }
        }

        if(mid <= 0. and not smallDistAtZero){
            if(current_repetition == 2){
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index + (*curve1_size_p)*(*curve2_size_p))] = 1.;
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index + (*curve1_size_p)*(*curve2_size_p))+1] = 0.;
                return;
            }else{
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index )] = 1.;
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index )+1] = 0.; 
                current_repetition++;      
                continue;
            }
        }

        if(mid >= 1. and not smallDistAtOne){
            if(current_repetition == 2){
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index + (*curve1_size_p)*(*curve2_size_p))] = 1.;
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index + (*curve1_size_p)*(*curve2_size_p))+1] = 0.;
                return;
            }else{
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index)] = 1.;
                results_p[2*(curve2_index*(*curve1_size_p) + curve1_index)+1] = 0.;  
                current_repetition++;     
                continue;
            }
        }

        if(discriminant < 0.){
            discriminant = 0.;
        }

        sqrt_discr = 0.;
        
        if(smallDistAtZero){
            begin = 0.;
        }else{
            sqrt_discr = (distance_t)sqrtf(discriminant);
            sqrt_discr_computed = true;

            const distance_t lambda1 = mid - sqrt_discr;
            const distance_t outershift = lambda1 - save_eps_half;
            distance_t innershift;
            if(1. < mid){
                if(lambda1 + save_eps_half< 1.){
                    innershift = lambda1 + save_eps_half;
                }else{
                    innershift = 1.;
                }
            }else{
                if(lambda1 + save_eps_half <mid){
                    innershift = lambda1 + save_eps_half;
                }else{
                    innershift = mid;
                }
            }

            bool small_dist_at_innershift;
            bool small_dist_at_outershift;
            distance_t circle_center_dist_innershift_sqr = 0;
            distance_t circle_center_dist1_outershift_sqr = 0;
            for(curve_size_t coordinate = 0; coordinate < *point_dimensions_p; coordinate++){
                distance_t coordinate_for_dist_innershift = line_start_p[coordinate] * (1. - innershift) + line_end_p[coordinate] * innershift;
                distance_t coordinate_for_dist1_outershift = line_start_p[coordinate] * (1. - outershift) + line_end_p[coordinate] * outershift;
                circle_center_dist_innershift_sqr += powf(circle_center_p[coordinate] - coordinate_for_dist_innershift, 2);
                circle_center_dist1_outershift_sqr += powf(circle_center_p[coordinate] - coordinate_for_dist1_outershift, 2);
                if(coordinate == *point_dimensions_p - 1){
                    small_dist_at_innershift = circle_center_dist_innershift_sqr <= radius_sqr;
                    small_dist_at_outershift = circle_center_dist1_outershift_sqr <= radius_sqr;
                }         
            }

            if(innershift >= outershift and small_dist_at_innershift and not small_dist_at_outershift){
                begin = innershift;
            }else{
                distance_t left = 0., right;
                if(mid < 1.){
                    right = mid;
                }else{
                    right = 1;
                }

                distance_t m = 0.5 * (left + right);
                while(right - left > save_eps){
                    m = 0.5 * (left + right);

                    bool small_dist_at_m;
                    distance_t circle_center_dist_m_sqr = 0;

                    for(curve_size_t coordinate = 0; coordinate < *point_dimensions_p; coordinate++){
                        distance_t coordinate_for_dist_m = line_start_p[coordinate] * (1. - m) + line_end_p[coordinate] * m;
                        circle_center_dist_m_sqr += powf(circle_center_p[coordinate] - coordinate_for_dist_m, 2);
                        if(coordinate == *point_dimensions_p - 1){
                            small_dist_at_m = circle_center_dist_m_sqr <= radius_sqr;
                        }         
                    }

                    if(small_dist_at_m){
                        right = m;
                    }else{
                        left = m;
                    }
                }
                begin = right;
            }
        }
        if(smallDistAtOne){
            end = 1.;
        }else{
            if(not sqrt_discr_computed){
                sqrt_discr = sqrtf(discriminant);
            }
            const distance_t lambda2 = mid + sqrt_discr;
            const distance_t outershift = lambda2 + save_eps_half;
            distance_t innershift;

            if(0 > mid){
                if( 0 > lambda2 - save_eps_half){
                    innershift = 0;
                }else{
                    innershift = lambda2 - save_eps_half;
                }
            }else{
                if( mid > lambda2 - save_eps_half){
                    innershift = mid;
                }else{
                    innershift = lambda2 - save_eps_half;
                }
            }

            bool small_dist_at_innershift;
            bool small_dist_at_outershift;
            distance_t circle_center_dist_innershift_sqr = 0;
            distance_t circle_center_dist1_outershift_sqr = 0;

            for(curve_size_t coordinate = 0; coordinate < *point_dimensions_p; coordinate++){
                distance_t coordinate_for_dist_innershift = line_start_p[coordinate] * (1. - innershift) + line_end_p[coordinate] * innershift;
                distance_t coordinate_for_dist1_outershift = line_start_p[coordinate] * (1. - outershift) + line_end_p[coordinate] * outershift;
                circle_center_dist_innershift_sqr += powf(circle_center_p[coordinate] - coordinate_for_dist_innershift, 2);
                circle_center_dist1_outershift_sqr += powf(circle_center_p[coordinate] - coordinate_for_dist1_outershift, 2);
                if(coordinate == *point_dimensions_p - 1){
                    small_dist_at_innershift = circle_center_dist_innershift_sqr <= radius_sqr;
                    small_dist_at_outershift = circle_center_dist1_outershift_sqr <= radius_sqr;
                }         
            }
            if(innershift <= outershift and small_dist_at_innershift and not small_dist_at_outershift){
                end = innershift;
            }else{
                distance_t left, right = 1.;
                if(mid > 0.){
                    left = mid;
                }else{
                    left = 0.;
                }
                distance_t m = 0.5 * (left + right);
                while(right - left > save_eps){
                    m = 0.5 * (left + right);

                    bool small_dist_at_m;
                    distance_t circle_center_dist_m_sqr = 0;

                    for(curve_size_t coordinate = 0; coordinate < *point_dimensions_p; coordinate++){
                        distance_t coordinate_for_dist_m = line_start_p[coordinate] * (1. - m) + line_end_p[coordinate] * m;
                        circle_center_dist_m_sqr += powf(circle_center_p[coordinate] - coordinate_for_dist_m, 2);
                        if(coordinate == *point_dimensions_p - 1){
                            small_dist_at_m = circle_center_dist_m_sqr <= radius_sqr;
                        }         
                    }

                    if(small_dist_at_m){
                        left = m;
                    }else{
                        right = m;
                    }
                }
                end = left;
            }
        }
        if(current_repetition == 2){
            results_p[2*(curve2_index*(*curve1_size_p) + curve1_index + (*curve1_size_p)*(*curve2_size_p))] = begin;
            results_p[2*(curve2_index*(*curve1_size_p) + curve1_index + (*curve1_size_p)*(*curve2_size_p))+1] = end;
            return;
        }else{
            results_p[2*(curve2_index*(*curve1_size_p) + curve1_index )] = begin;
            results_p[2*(curve2_index*(*curve1_size_p) + curve1_index )+1] = end;
        }
            current_repetition++;
    }
}

cudaError_t Cuda_intersection::intersection_interval_call_gpu(
    distance_t radius
){
    cudaError_t cudaStatus_global = cudaSuccess;

    for(int device_nbr = 0; device_nbr < number_devices; device_nbr++){
        //Cuda launching utils
        cudaError_t cudaStatus;
        curve_size_t number_of_threads = curve1_size[device_nbr] * curve2_size[device_nbr];
        curve_size_t number_of_blocks = 0;

        cudaStatus = cudaSetDevice(device_nbr);
        if(cudaStatus != cudaSuccess){
            std::cerr << "CUDASetDevice failed! Do you have CUDA-capable GPU installed?" << std::endl;
        }
    
        cudaStatus = cudaMemcpy(dev_radius_p[device_nbr], &radius, sizeof(distance_t), cudaMemcpyHostToDevice );
        if (cudaStatus != cudaSuccess ){
            std::cerr << "CudaMemcpy radius to dev_radius_p failed!" << std::endl;
            cudaStatus_global = cudaStatus;
            goto Error;
        }

        if((number_of_threads%BLOCKSIZE) == 0){
            number_of_blocks = number_of_threads / BLOCKSIZE;
        }else{
            number_of_blocks = ((curve_size_t)(number_of_threads/ BLOCKSIZE)) + 1;
        }
        cuda_kernel1_calculate_intervals <<<number_of_blocks, BLOCKSIZE >>>(    dev_results_p[device_nbr], 
                                                                                dev_points_curve1_p[device_nbr], 
                                                                                dev_points_curve2_p[device_nbr],
                                                                                dev_curve1_size_p[device_nbr],
                                                                                dev_curve2_size_p[device_nbr],
                                                                                dev_point_dimensions_p[device_nbr],
                                                                                dev_radius_p[device_nbr],
                                                                                dev_is_last_kernel_p[device_nbr],
                                                                                eps
        );
    }
    
    for(int device_nbr = 0; device_nbr < number_devices; device_nbr++){
        cudaError_t cudaStatus;

        cudaStatus = cudaSetDevice(device_nbr);
        if(cudaStatus != cudaSuccess){
            std::cerr << "CUDASetDevice failed! Do you have CUDA-capable GPU installed?" << std::endl;
        }

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess){
            std::cerr << "CudaGetLastError returned error code: " << cudaStatus << " after launching kernel!" << std::endl;
            std::cerr << cudaGetErrorString(cudaStatus) << std::endl;
            cudaStatus_global = cudaStatus;
            goto Error;  
        }

        // Waits until all threads are done with their job.
        cudaStatus = cudaDeviceSynchronize();
        if( cudaStatus != cudaSuccess){
            std::cerr << "CudaDeviceSynchronize() returned error code: " << cudaStatus << " after launching kernel!" << std::endl;
            std::cerr << cudaGetErrorString(cudaStatus) << std::endl;
            cudaStatus_global = cudaStatus;
            goto Error;  
        }

        if(device_nbr == number_devices - 1){
            cudaStatus = cudaMemcpy(
                host_results_p + 2* (curve1_end_index[number_devices-1]+1)*(curve2_start_index[device_nbr]), 
                dev_results_p[device_nbr], sizeof(distance_t) * (2 * curve1_size[device_nbr] * curve2_size[device_nbr]), 
                cudaMemcpyDeviceToHost
            );
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMemcpy dev_results into results failed!" << std::endl;
                cudaStatus_global = cudaStatus;
                goto Error;
            }

            cudaStatus = cudaMemcpy(
                 host_results_p +2* (curve1_end_index[number_devices-1]+1)*(curve2_end_index[number_devices-1]+1) +2*(curve1_end_index[number_devices-1]+1)*(curve2_start_index[device_nbr]), 
                 dev_results_p[device_nbr] + 2*(curve1_size[device_nbr])*(curve2_size[device_nbr]),
                 sizeof(distance_t) * (2 * curve1_size[device_nbr] * curve2_size[device_nbr]), 
                 cudaMemcpyDeviceToHost
            );
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMemcpy dev_results into results failed!" << std::endl;
                cudaStatus_global = cudaStatus;
                goto Error;
            }
        }else{
            cudaStatus = cudaMemcpy(
                host_results_p + 2*(curve1_end_index[number_devices-1]+1)*(curve2_start_index[device_nbr]), 
                dev_results_p[device_nbr],
                sizeof(distance_t) * (2 * curve1_size[device_nbr] * curve2_size[device_nbr]), 
                cudaMemcpyDeviceToHost
            );
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMemcpy dev_results into results failed!" << std::endl;
                cudaStatus_global = cudaStatus;
                goto Error;
            }

            cudaStatus = cudaMemcpy(
                host_results_p + 2*(curve1_end_index[number_devices-1]+1)*(curve2_end_index[number_devices-1]+1) +2*(curve1_end_index[number_devices-1]+1)*(curve2_start_index[device_nbr]), 
                dev_results_p[device_nbr] + 2*(curve1_size[device_nbr])*(curve2_size[device_nbr]),
                sizeof(distance_t) * (2 * curve1_size[device_nbr] * curve2_size[device_nbr]), 
                cudaMemcpyDeviceToHost
            );
            if(cudaStatus != cudaSuccess){
                std::cerr << "CudaMemcpy dev_results into results failed!" << std::endl;
                cudaStatus_global = cudaStatus;
                goto Error;
            }
        } 
    }

    goto Regular_finish;

    //Free the memory on Error
    Error:
        free_memory();
        return cudaStatus_global;
    
    Regular_finish:
        return cudaStatus_global;
}
