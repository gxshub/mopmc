if(EXISTS "/home/thomas/postdoc/mopmc-dev/cmake-build-default/ProjectTests[1]_tests.cmake")
  include("/home/thomas/postdoc/mopmc-dev/cmake-build-default/ProjectTests[1]_tests.cmake")
else()
  add_test(ProjectTests_NOT_BUILT ProjectTests_NOT_BUILT)
endif()
