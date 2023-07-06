##===----------------------------------------------------------------------===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# Build a plugin for a EPI accelerator if available.
#
##===----------------------------------------------------------------------===##
if (NOT(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building EPI offloading plugin: only support EPI in Linux x86_64 or aarch64 hosts.")
  return()
elseif (NOT LIBOMPTARGET_DEP_LIBELF_FOUND)
  libomptarget_say("Not building EPI offloading plugin: libelf dependency not found.")
  return()
endif()

libomptarget_say("Building EPI offloading plugin.")

# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=EPI)


add_library(omptarget.rtl.epi SHARED src/rtl.cpp)

# Install plugin under the lib destination folder.
install(TARGETS omptarget.rtl.epi LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")

target_link_libraries(omptarget.rtl.epi
  ${LIBOMPTARGET_DEP_EPI_DRIVER_LIBRARIES}
  ${LIBOMPTARGET_DEP_LIBELF_LIBRARIES}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports")

target_include_directories(omptarget.rtl.epi PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS}
)

# Report to the parent scope that we are building a plugin for EPI.
set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} riscv64-unknown-linux-gnu" PARENT_SCOPE)
