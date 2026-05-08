cmake_minimum_required(VERSION 3.21)
project(TriLibGo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/Zc:__cplusplus)
endif()

option(TRILIBGO_BUILD_APP "Build the Qt demo application" ON)
option(TRILIBGO_BUILD_TESTS "Build tests" ON)
option(TRILIBGO_ENABLE_ONNXRUNTIME "Enable ONNX Runtime integration when SDK is available" ON)

add_library(trilibgo_core
    src/core/board_topology.cpp
    src/core/game_state.cpp
    src/core/rules_engine.cpp
    src/core/game_record.cpp
)
target_include_directories(trilibgo_core PUBLIC include)

add_library(trilibgo_ai
    src/ai/analysis_model.cpp
    src/ai/feature_encoder.cpp
    src/ai/onnx_analysis_model.cpp
    src/ai/random_agent.cpp
)
target_include_directories(trilibgo_ai PUBLIC include)
target_link_libraries(trilibgo_ai PUBLIC trilibgo_core)

if(TRILIBGO_ENABLE_ONNXRUNTIME AND DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    if(WIN32)
        set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    elseif(APPLE)
        set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
    else()
        set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    endif()
    if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h" AND EXISTS "${ONNXRUNTIME_LIBRARY}")
        target_include_directories(trilibgo_ai PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
        target_link_libraries(trilibgo_ai PRIVATE "${ONNXRUNTIME_LIBRARY}")
        target_compile_definitions(trilibgo_ai PRIVATE TRILIBGO_HAS_ONNXRUNTIME=1)
    endif()
endif()

if(TRILIBGO_BUILD_TESTS)
    enable_testing()
    add_executable(trilibgo_tests tests/core_tests.cpp)
    target_link_libraries(trilibgo_tests PRIVATE trilibgo_core trilibgo_ai)
    add_test(NAME trilibgo_core_tests COMMAND trilibgo_tests)
endif()

if(TRILIBGO_BUILD_APP)
    find_package(Qt6 REQUIRED COMPONENTS Widgets)
    qt_standard_project_setup()

    qt_add_executable(trilibgo_app
        include/trilibgo/app/main_window.h
        include/trilibgo/app/board_widget.h
        src/app/main.cpp
        src/app/main_window.cpp
        src/app/board_widget.cpp
        src/app/game_controller.cpp
        src/app/replay_compat.cpp
    )
    target_include_directories(trilibgo_app PRIVATE include)
    target_link_libraries(trilibgo_app PRIVATE Qt6::Widgets trilibgo_core trilibgo_ai)
endif()
