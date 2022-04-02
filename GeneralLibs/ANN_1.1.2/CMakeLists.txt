PROJECT(ann)

ADD_DEFINITIONS( 
  -DANN_NO_RANDOM
  -DANN_PERF
  -DDLL_EXPORTS
#  -DWIN32
#  -D_WINDOWS
)

SET(${PROJECT_NAME}_SRCS
  src/ANN.cpp
  src/bd_fix_rad_search.cpp
  src/bd_pr_search.cpp
  src/bd_search.cpp
  src/bd_tree.cpp
  src/brute.cpp
  src/kd_dump.cpp
  src/kd_fix_rad_search.cpp
  src/kd_pr_search.cpp
  src/kd_search.cpp
  src/kd_split.cpp
  src/kd_tree.cpp
  src/kd_util.cpp
  src/perf.cpp
)

INCLUDE_DIRECTORIES(include)

ADD_LIBRARY(${PROJECT_NAME} ${${PROJECT_NAME}_SRCS})
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES FOLDER GeneralLibs)

