cmake_minimum_required(VERSION 3.16)
project(imder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_AUTOMOC ON)

find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(imder src/imder.cpp)
if(WIN32)
  target_sources(imder PRIVATE app.rc)
endif()

target_link_libraries(imder PRIVATE Qt5::Core Qt5::Gui Qt5::Widgets ${OpenCV_LIBS})

if(MSVC)
  target_compile_options(imder PRIVATE /EHsc /utf-8)
  target_link_options(imder PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)
else()
  target_compile_options(imder PRIVATE -O2)
endif()

install(TARGETS imder RUNTIME DESTINATION .)
