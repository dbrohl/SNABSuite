/*
 *  SNABSuite -- Spiking Neural Architecture Benchmark Suite
 *  Copyright (C) 2016  Christoph Jenzen
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <glob.h>

#include <cypress/cypress.hpp>

#include "common/benchmark.hpp"
#include "energy/energy_recorder.hpp"
#include "energy/energy_utils.hpp"

using namespace SNAB;

// Compatibility hack for for older glibc
__asm__(".symver glob64,glob64@GLIBC_2.2.5");

int main(int argc, const char *argv[])
{
	if ((argc < 2 || argc > 5) && !cypress::NMPI::check_args(argc, argv)) {
		std::cout << "Usage: " << argv[0]
		          << " <SIMULATOR> [snab] [bench_index] [NMPI]" << std::endl;
		return 1;
	}

	if (std::string(argv[argc - 1]) == "NMPI" &&
	    !cypress::NMPI::check_args(argc, argv)) {
		glob_t glob_result;
		glob(std::string("../config/*").c_str(), GLOB_TILDE, NULL,
		     &glob_result);
		std::vector<std::string> files;

		for (unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
			files.push_back(std::string(glob_result.gl_pathv[i]));
		}
		globfree(&glob_result);
		cypress::NMPI(argv[1], argc, argv, files, true);
		return 0;
	}

	size_t bench_index = 0;
	if (isdigit(*argv[argc - 1])) {
		bench_index = std::stoi(argv[argc - 1]);
	}

	cypress::global_logger().min_level(cypress::LogSeverity::INFO, 1);

	std::string snab_name = "all";
	if (argc > 2 && std::string(argv[2]) != "NMPI" && !isdigit(*argv[2])) {
		snab_name = std::string(argv[2]);
	}

	// Convert names
	std::string simulator = argv[1];
	if (simulator == "spiNNaker") {
		simulator = "spinnaker";
	}
	else if (simulator == "hardware.hbp_pm") {
		simulator = "nmpm1";
	}
	else if (simulator == "nest") {
		simulator = "pynn.nest";
	}

	/*auto multi = std::make_shared<Energy::Multimeter>("", 0, true);
	multi->set_block(true);
	multi->start_recording();*/

	BenchmarkExec bench(std::string(argv[1]), snab_name, bench_index);

	/*multi->stop_recording();
	std::cout << "Average Power Draw in W "
	          << multi->average_power_draw() / 1000.0 << std::endl;
	std::cout << "Energy in J " << multi->calculate_energy() / 1000.0
	          << std::endl;*/
	/*auto min = multi->min_current();
	auto thresh = min + ((multi->max_current() - min) * 0.9);
	multi->average_power_draw_last(thresh) / 1000.0;*/

	return 0;
}
