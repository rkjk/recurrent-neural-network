cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project(recurrent)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package( Threads REQUIRED)

if(CMAKE_USE_PTHREADS_INIT)
      set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-pthread")
endif()

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(SOURCES 
    ${SRC_DIR}
    ${SRC_DIR}/rnn.cc 
    ${SRC_DIR}/rnn_neuron_functions.cc
    ${SRC_DIR}/rnn_main.cc
)

set(HEADERS
    ${SRC_DIR}
    ${SRC_DIR}/rnn_headers.h
)



set(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC "${SRC_DIR}")

target_link_libraries( ${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} )
