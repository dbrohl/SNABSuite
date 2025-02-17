#  SNABSuite -- Spiking Neural Architecture Benchmark Suite
#  Copyright (C) 2020 Christoph Ostrau
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

include(FetchContent)
FetchContent_Declare(sudoku_rep
    GIT_REPOSITORY      "https://github.com/hbp-unibi/SpikingSudokuSolver"
    GIT_TAG             master
)
FetchContent_GetProperties(sudoku_rep)
if(NOT sudoku_rep_POPULATED)
    FetchContent_Populate(sudoku_rep)
endif()
include_directories(
	${sudoku_rep_SOURCE_DIR}/
	${sudoku_rep_SOURCE_DIR}/source/
)
add_library(sudoku
    ${sudoku_rep_SOURCE_DIR}/source/utils/Sudoku
    ${sudoku_rep_SOURCE_DIR}/source/utils/spikingSudokuSolver
)

# target_link_libraries(sudoku
# 	${CYPRESS_LIBRARY}
# 	-pthread
# )
add_dependencies(sudoku cypress_ext)
