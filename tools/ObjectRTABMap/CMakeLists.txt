
SET(INCLUDE_DIRS
	${PROJECT_SOURCE_DIR}/corelib/include
	${PROJECT_SOURCE_DIR}/utilite/include
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

SET(LIBRARIES
	${OpenCV_LIBS} 
	${PCL_LIBRARIES} 
	${ZLIB_LIBRARIES} 
)

####################################
# Generate resources files END
####################################

add_definitions(${PCL_DEFINITIONS})


# Make sure the compiler can find include files from our library.
INCLUDE_DIRECTORIES(${INCLUDE_DIRS})

# Add binary that is built from the source file "main.cpp".
# The extension is automatically found.
#ADD_LIBRARY(rtabmap_core ${SRC_FILES} ${RESOURCES_HEADERS})
ADD_EXECUTABLE(object_rtabmap main.cpp)
TARGET_LINK_LIBRARIES(object_rtabmap rtabmap_core rtabmap_utilite ${LIBRARIES})

SET_TARGET_PROPERTIES( object_rtabmap
  PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-object_rtabmap)

INSTALL(TARGETS object_rtabmap
		RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
		BUNDLE DESTINATION "${CMAKE_BUNDLE_LOCATION}" COMPONENT runtime)
		
