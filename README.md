This is an implementation of the **continuous** Fréchet distance, parallelized using Nvidia CUDA.

# Set up the enviroment:
The steps listed below are all the steps you need to get the code running on a new ubuntu 18.04 installation.
Most users will already have most necessary package installed, but be careful
your packages might be outdated and that can lead to unexpected errors.
1) Install python3.6
2) install pip3
3) "pip3 install boost"     
4) Install Cuda if you have a cuda capable gpu:  
  * Remove old nvidia drivers:   
    * sudo apt purge nvidia*  
  * Add repository of drivers:   
    * sudo add-apt-repository ppa:graphics-drivers  
  * Update:   
    * sudo apt update  
  * Install a driver:   
    * sudo apt-get install nvidia-390  
  * Install cuda:   
    * sudo apt-get install nvidia-cuda-toolkit  
  * Install dependencies to use cuda:   
    * sudo apt-get install gcc-6 g++-6 g++-6-multilib gfortran-6  
5) Install the latest version of cmake (Version 3.14 worked just fine)
6) Go to the root folder:
  * "make pre"
  * "make python3"
7) Go to the data folder: 
  * chmod +x download_data.sh
  * ./download_data.sh
8) Go to the py folder and run the code:
  * python3 <file>
			
			
			
The implementation is based on André Nusser's implementation: https://github.com/chaot4/frechet_distance

