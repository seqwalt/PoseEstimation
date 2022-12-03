Compile using:
g++ -std=c++11 -o example optical_flow.cpp `pkg-config --libs opencv4` `pkg-config --cflags opencv4`

To run:
./example testvid1.mov 