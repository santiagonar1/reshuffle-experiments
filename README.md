# Modern C++ Template

This is a template that can be used to start developing a project in C++. We use [conan](https://conan.io/) as
package manager.

## Build and run

We need some additional tools in order to build it (besides the usual C++ compiler). In particular,
please make sure to install:

- Conan 2. You can find the official documentation [here](https://docs.conan.io/2/installation.html).
- CMake and Make. For example in an Ubuntu system, the latter can be installed simply with `sudo apt install make`.

Once both of them are installed, you simply need to execute:

```shell
mkdir build && cd build
cmake -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=conan_provider.cmake ..
make
```

Now, you should find the binary inside `build/app/`.

## Docker

You can also use the [cpp-dev](https://github.com/santiagonar1/dockerfiles/blob/main/cpp-dev.dockerfile) docker image
to compile and run the code. If you do not want to build it yourself, run:

```shell
./run-docker.sh
```