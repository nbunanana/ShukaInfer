cmake_minimum_required(VERSION 3.16)
project(shukainfer)

# 指定cpp版本(必须为cpp17否则报错)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# 定义PROJECT_ROOT，值为当前CmakeLists文件路径
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# 生成 compile_commands.json（记录了项目中每个源文件是如何被编译的，包括完整的编译命令、编译器路径、头文件搜索路径、宏定义等），供 clangd 等开发工具使用。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

 # 只有用 MSVC 编译时才执行里面的内容（GCC/Clang 完全跳过）
if(MSVC)
    # 假如编译x64，那么SSE2永远可用
    # 假如编译x86，那么SSE2可用的话，就开启SSE2
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if (CMAKE_SIZEOF_VOID_P EQUAL 4)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("/arch:SSE2" HAS_SSE2)
    else()
        set(HAS_SSE2 ON)
    endif()
    if(HAS_SSE2)
        # # set __SSE2__ and __XOP__ macros
        add_definitions(-D__SSE2__ -D__XOP__)
    endif()
    # Force LLVM OpenMP on MSVC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm /openmp:experimental /arch:AVX2")
endif()

option(BUILD_DEMO "BUILD THE DEMO PROJECT")
if(BUILD_DEMO)
    MESSAGE(STATUS "BUILD DEMO PROJECT" OFF)
    add_subdirectory(demos)
endif()

# 添加头文件寻找路径
include_directories(./include)

# 找需要使用的包
find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# 添加源文件,aux_source_directory(<目录> <变量名>)是添加目录下面所有的源文件
aux_source_directory(./source/data DIR_DATA)

aux_source_directory(./source/runtime DIR_RUNTIME)
aux_source_directory(./source/runtime/pnnx DIR_RUNTIME)

aux_source_directory(./source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(./source/layer/details DIR_BINOCULAR_LAYER)
aux_source_directory(./source/parser DIR_PARSER)
aux_source_directory(./source/utils/time DIR_TIME_UTILS)
aux_source_directory(./source/utils/math DIR_MATH_UTILS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# 设置编译产物输出位置
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 生成连接清单
set(link_lib glog::glog)
# 不是win系统把pthread也加进清单
if(!WIN32)
    set(link_lib ${link_lib} pthread)
endif()

set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# 编译出一个库
add_library(Shuka SHARED ${DIR_DATA} ${DIR_RUNTIME} 
                         ${DIR_ABSTRACT_LAYER} ${DIR_BINOCULAR_LAYER} 
                         ${DIR_PARSER} ${DIR_TIME_UTILS} 
                         ${DIR_MATH_UTILS}
)
target_link_libraries(Shuka ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

# mathfun library defines
add_compile_definitions(SSE_MATHFUN_WITH_CODE USE_SSE_AUTO)


# 开发模式
option(DEVELOPMENT ON)
if(${DEVELOPMENT})
    message(STATUS "DEVELOPMENT MODE ON")
    enable_testing()
    # add_subdirectory(bench)
    add_subdirectory(test)
endif()
