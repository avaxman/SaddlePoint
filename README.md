# SaddlePoint
SaddlePoint is a templated header-only optimization package based on Eigen, which is meant for configurable optimization. The idea is that one could plug-in concept classes that change much of the behaviour of the optimizer. For instance, the linear solver itself.

To make and  the benchmak example, do

```
cd benchmark/EXAMPLE
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./EXAMPLE_bin
```

Send any complaints, suggestions, or any other helpful advice to Amir Vaxman (<avaxman@gmail.com>)
