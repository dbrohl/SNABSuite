/*
 *  SNABSuite - Spiking Neural Architecture Benchmark Suite
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

#include <cypress/cypress.hpp>  // Avoid a warning
#include <memory>
#include <string>
#include <vector>

#include "SNABs/activation_curve.hpp"
#include "SNABs/binam.hpp"
#include "SNABs/function.hpp"
#include "SNABs/max_input.hpp"
#include "SNABs/max_inter_neuron.hpp"
#include "SNABs/mnist/mnist.hpp"
#include "SNABs/output_bench.hpp"
#include "SNABs/refractory_period.hpp"
#include "SNABs/setup_time.hpp"
#include "SNABs/slam.hpp"
#include "SNABs/sudoku.hpp"
#include "SNABs/wta_like.hpp"
#include "common/snab_base.hpp"
#include "common/snab_registry.hpp"

namespace SNAB {
std::vector<std::shared_ptr<SNABBase>> snab_registry(std::string backend,
                                                     size_t bench_index)
{
	std::vector<std::shared_ptr<SNABBase>> vec = {
	    std::make_shared<OutputFrequencySingleNeuron>(
	        OutputFrequencySingleNeuron(backend, bench_index)),
	    std::make_shared<OutputFrequencySingleNeuron2>(
	        OutputFrequencySingleNeuron2(backend, bench_index)),
	    std::make_shared<OutputFrequencyMultipleNeurons>(
	        OutputFrequencyMultipleNeurons(backend, bench_index)),
	    std::make_shared<RefractoryPeriod>(
	        RefractoryPeriod(backend, bench_index)),
	    std::make_shared<MaxInputOneToOne>(
	        MaxInputOneToOne(backend, bench_index)),
	    std::make_shared<MaxInputAllToAll>(
	        MaxInputAllToAll(backend, bench_index)),
	    std::make_shared<MaxInputFixedOutConnector>(
	        MaxInputFixedOutConnector(backend, bench_index)),
	    std::make_shared<MaxInputFixedInConnector>(
	        MaxInputFixedInConnector(backend, bench_index)),
	    std::make_shared<SingleMaxFreqToGroup>(
	        SingleMaxFreqToGroup(backend, bench_index)),
	    std::make_shared<GroupMaxFreqToGroup>(
	        GroupMaxFreqToGroup(backend, bench_index)),
	    std::make_shared<GroupMaxFreqToGroupAllToAll>(
	        GroupMaxFreqToGroupAllToAll(backend, bench_index)),
	    std::make_shared<GroupMaxFreqToGroupProb>(
	        GroupMaxFreqToGroupProb(backend, bench_index)),
	    std::make_shared<SetupTimeOneToOne>(
	        SetupTimeOneToOne(backend, bench_index)),
	    std::make_shared<SetupTimeAllToAll>(
	        SetupTimeAllToAll(backend, bench_index)),
	    std::make_shared<SetupTimeRandom>(
	        SetupTimeRandom(backend, bench_index)),
	    std::make_shared<SimpleWTA>(SimpleWTA(backend, bench_index)),
	    std::make_shared<LateralInhibWTA>(
	        LateralInhibWTA(backend, bench_index)),
	    std::make_shared<MirrorInhibWTA>(MirrorInhibWTA(backend, bench_index)),
	    std::make_shared<MirrorInhibWTASmall>(MirrorInhibWTASmall(backend, bench_index)),
	    std::make_shared<WeightDependentActivation>(
	        WeightDependentActivation(backend, bench_index)),
	    std::make_shared<RateBasedWeightDependentActivation>(
	        RateBasedWeightDependentActivation(backend, bench_index)),
	    std::make_shared<ReluSimilarity>(
	        ReluSimilarity(backend, bench_index)),
	    std::make_shared<MnistSpikey>(MnistSpikey(backend, bench_index)),
	    std::make_shared<MnistNAS63>(MnistNAS63(backend, bench_index)),
	    std::make_shared<MnistNAS129>(MnistNAS129(backend, bench_index)),
	    std::make_shared<MnistNAStop>(MnistNAStop(backend, bench_index)),
	    std::make_shared<MnistDiehl>(MnistDiehl(backend, bench_index)),
	    std::make_shared<MnistITLLastLayer>(
	        MnistITLLastLayer(backend, bench_index)),
	    std::make_shared<MnistITL>(
	        MnistITL(backend, bench_index)),
	    std::make_shared<MnistSpikeyTTFS>(
	        MnistSpikeyTTFS(backend, bench_index)),
	    std::make_shared<MnistDiehlTTFS>(
	        MnistDiehlTTFS(backend, bench_index)),
	    std::make_shared<MnistITLTTFS>(
	        MnistITLTTFS(backend, bench_index)),
	    std::make_shared<BiNAM>(BiNAM(backend, bench_index)),
	    std::make_shared<BiNAM_pop>(BiNAM_pop(backend, bench_index)),
	    std::make_shared<BiNAM_burst>(BiNAM_burst(backend, bench_index)),
	    std::make_shared<BiNAM_pop_burst>(
	        BiNAM_pop_burst(backend, bench_index)),
	    std::make_shared<SpikingSudoku>(
	        SpikingSudoku(backend, bench_index)),
	    std::make_shared<SpikingSudokuSinglePop>(
	        SpikingSudokuSinglePop(backend, bench_index)),
	    std::make_shared<SpikingSudokuMirrorInhib>(
	        SpikingSudokuMirrorInhib(backend, bench_index)),
	    std::make_shared<SpikingSlam>(
	        SpikingSlam(backend, bench_index)),
	    std::make_shared<FunctionApproximation>(
	        FunctionApproximation(backend, bench_index)),
	    std::make_shared<MnistDoubleCNN>(
	        MnistDoubleCNN(backend, bench_index)),
	    std::make_shared<MnistCNNPool>(
	        MnistCNNPool(backend, bench_index)),
	};
	return vec;
}
}  // namespace SNAB
