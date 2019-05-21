all:
	cd build && cmake ..
	cd build && make

pre:
	sudo apt install libboost-all-dev
	sudo apt-get install python3-setuptools
	sudo apt-get install python3-numpy
	sudo apt-get install python3-pandas
	sudo apt-get install python-setuptools
	sudo apt-get install python-numpy
	sudo apt-get install python-pandas

python3:
	cd py && python3 ./setup.py install --user
	
python2:
	cd py && python2 ./setup.py install --user
