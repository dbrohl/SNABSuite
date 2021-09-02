/*
 *  SNABSuite -- Spiking Neural Architecture Benchmark Suite
 *  Copyright (C) 2019  Christoph Ostrau
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

#include "mnist.hpp"

#include <cypress/backend/power/power.hpp>
#include <cypress/cypress.hpp>  // Neural network frontend
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "mnist_mlp.hpp"
#include "util/utilities.hpp"

namespace SNAB {
using namespace cypress;
MNIST_BASE::MNIST_BASE(const std::string backend, size_t bench_index)
    : MNIST_BASE(backend, bench_index, __func__)
{
}

MNIST_BASE::MNIST_BASE(const std::string backend, size_t bench_index,
                       std::string name)
    : SNABBase(name, backend, {"accuracy", "sim_time", "response time"},
               {"quality", "performance", "performance"},
               {"accuracy", "time", "time"}, {"", "s", "ms"},
               {"neuron_type", "neuron_params", "images", "batchsize",
                "duration", "max_freq", "pause", "poisson", "max_weight",
                "train_data", "batch_parallel", "dnn_file", "scaled_image"},
               bench_index)
{
}

void MNIST_BASE::read_config()
{
	m_neuron_type_str = m_config_file["neuron_type"].get<std::string>();

	m_neuro_params =
	    NeuronParameter(SpikingUtils::detect_type(m_neuron_type_str),
	                    m_config_file["neuron_params"]);
	if (m_neuron_type_str == "IF_cond_exp") {
		m_neuro_params.set("tau_syn_I", m_neuro_params.get("tau_syn_E"));
	}
	m_images = m_config_file["images"].get<size_t>();
	m_batchsize = m_config_file["batchsize"].get<size_t>();
	m_duration = m_config_file["duration"].get<Real>();
	m_max_freq = m_config_file["max_freq"].get<Real>();
	m_pause = m_config_file["pause"].get<Real>();
	m_poisson = m_config_file["poisson"].get<bool>();
	m_max_weight = m_config_file["max_weight"].get<Real>();
	m_max_pool_weight = m_config_file["max_pool_weight"].empty() ? 0.1 : m_config_file["pool_max_weight"].get<Real>();
	m_pool_inhib_weight = m_config_file["pool_inhib_weight"].empty() ? -0.1 : m_config_file["pool_inhib_weight"].get<Real>();
	m_pool_delay = m_config_file["pool_delay"].empty() ? 0.3 : m_config_file["pool_delay"].get<Real>();
	m_train_data = m_config_file["train_data"].get<bool>();
	m_batch_parallel = m_config_file["batch_parallel"].get<bool>();
	m_dnn_file = m_config_file["dnn_file"].get<std::string>();
	m_scaled_image = m_config_file["scaled_image"].get<bool>();
	m_weights_scale_factor = 0.0;
	if (m_config_file.find("count_spikes") != m_config_file.end()) {
		m_count_spikes = m_config_file["count_spikes"].get<bool>();
	}
	if (m_config_file.find("gamma") != m_config_file.end())
	{
		m_gamma = m_config_file["gamma"].get<Real>();
	}


#if SNAB_DEBUG
	m_count_spikes = true;
#endif
	if (m_config_file.find("ttfs") != m_config_file.end()) {
		m_ttfs = m_config_file["ttfs"].get<bool>();
	}
	if (m_config_file.find("activity_based_scaling") != m_config_file.end()) {
		m_activity_based_scaling =
		    m_config_file["activity_based_scaling"].get<size_t>();
	}
}

cypress::Network &MNIST_BASE::build_netw_int(cypress::Network &netw)
{
	read_config();
	auto kerasdata = mnist_helper::read_network(m_dnn_file, true);
	m_mlp = std::make_shared<MNIST::MLP<>>(kerasdata, 0, m_batchsize, 0.0);
	if (m_scaled_image) {
		m_mlp->scale_down_images();
	}
	mnist_helper::SPIKING_MNIST spike_mnist;
	if (m_train_data) {
		spike_mnist = mnist_helper::mnist_to_spike(m_mlp->mnist_train_set(),
		                                           m_duration, m_max_freq,
		                                           m_images, m_poisson, m_ttfs);
	}
	else {
		spike_mnist = mnist_helper::mnist_to_spike(m_mlp->mnist_test_set(),
		                                           m_duration, m_max_freq,
		                                           m_images, m_poisson, m_ttfs);
	}
	m_batch_data = mnist_helper::create_batches(spike_mnist, m_batchsize,
	                                            m_duration, m_pause, false);

	if (m_activity_based_scaling) {
		m_layer_scale_factors =
		    m_mlp->rescale_weights(m_activity_based_scaling);
		std::string message;
		for (auto i : m_layer_scale_factors) {
			message += std::to_string(i);
			message += ", ";
		}
		global_logger().debug("SNABSuite", "SNN rescale factors: " + message);
	}

	m_label_pops.clear();
	m_networks.clear();
	m_all_pops.clear();
	if (m_batch_parallel) {
		for (auto &i : m_batch_data) {
			mnist_helper::create_spike_source(netw, i);
			create_deep_network(netw, m_max_weight,
			                    m_max_pool_weight, m_pool_inhib_weight);
			m_label_pops.emplace_back(netw.populations().back());
		}

		if (m_count_spikes) {
			for (auto pop : netw.populations()) {
				pop.signals().record(0);
				m_all_pops.emplace_back(pop);
			}
		}
	}
	else {
		for (auto &i : m_batch_data) {
			m_networks.push_back(cypress::Network());
			mnist_helper::create_spike_source(m_networks.back(), i);
			create_deep_network(m_networks.back(), m_max_weight,
			                    m_max_pool_weight, m_pool_inhib_weight);
			m_label_pops.emplace_back(m_networks.back().populations().back());
			if (m_count_spikes) {
				for (auto pop : m_networks.back().populations()) {
					pop.signals().record(0);
					m_all_pops.emplace_back(pop);
				}
			}
		}
	}

	for (auto &pop : m_label_pops) {
		pop.signals().record(0);
	}

#if SNAB_DEBUG
	Utilities::write_vector2_to_csv(std::get<0>(m_batch_data[0]),
	                                _debug_filename("spikes_input.csv"));
	Utilities::plot_spikes(_debug_filename("spikes_input.csv"), m_backend);
#endif
	return netw;
}

cypress::Network &MNIST_BASE::build_netw(cypress::Network &netw)
{
	return build_netw_int(netw);
}

void MNIST_BASE::run_netw(cypress::Network &netw)
{
	cypress::PowerManagementBackend pwbackend(
	    cypress::Network::make_backend(m_backend));
	try {

		if (m_batch_parallel) {
			netw.run(pwbackend, m_batchsize * (m_duration + m_pause));
		}
		else {
			global_logger().info(
			    "SNABSuite",
			    "batch not run in parallel, using internal network objects!");
			for (auto &pop : m_label_pops) {
				pop.network().run(pwbackend,
				                  m_batchsize * (m_duration + m_pause));
			}
		}
	}
	catch (const std::exception &exc) {
		std::cerr << exc.what();
		global_logger().fatal_error(
		    "SNABSuite",
		    "Wrong parameter setting or backend error! Simulation broke down");
	}
}
namespace {
std::vector<Real> TTFS_response_time(
    const std::vector<PopulationBase> &label_pops, size_t batch_size,
    Real duration, Real pause)
{

	std::vector<Real> time_to_sol;
	for (const auto &pop : label_pops) {

		std::vector<std::vector<Real>> binned_spike_times;
		for (const auto &neuron : pop) {
			binned_spike_times.push_back(SpikingUtils::spike_time_binning_TTFS(
			    0.0, Real(batch_size) * (duration + pause), batch_size,
			    neuron.signals().data(0)));
		}

		for (size_t sample = 0; sample < batch_size; sample++) {
			Real min = std::numeric_limits<Real>::max();
			for (size_t neuron = 0; neuron < binned_spike_times.size();
			     neuron++) {
				if (binned_spike_times[neuron][sample] < min) {
					min = binned_spike_times[neuron][sample];
				}
			}
			if (min < std::numeric_limits<Real>::max()) {
				min -= Real(sample) * (duration + pause);
				time_to_sol.emplace_back(min);
			}
		}
	}
	return time_to_sol;
}
}  // namespace

std::vector<std::array<cypress::Real, 4>> MNIST_BASE::evaluate()
{
	size_t global_correct(0);
	size_t images(0);
	for (size_t batch = 0; batch < m_label_pops.size(); batch++) {
		auto pop = m_label_pops[batch];
		auto labels = mnist_helper::spikes_to_labels(pop, m_duration, m_pause,
		                                             m_batchsize, m_ttfs);
		auto &orig_labels = std::get<1>(m_batch_data[batch]);
		auto correct = mnist_helper::compare_labels(orig_labels, labels);
		global_correct += correct;
		images += orig_labels.size();

#if SNAB_DEBUG
		std::cout << "Target\t Infer" << std::endl;
		for (size_t i = 0; i < orig_labels.size(); i++) {
			std::cout << orig_labels[i] << "\t" << labels[i] << std::endl;
		}
		if (m_batch_parallel && batch == 0) {
			std::vector<std::vector<cypress::Real>> spikes;
			auto pop2 = m_netw.populations()[0];
			for (size_t i = 0; i < pop2.size(); i++) {
				spikes.push_back(pop2[i].signals().data(0));
			}
			Utilities::write_vector2_to_csv(
			    spikes, _debug_filename("spikes_input_" +
			                            std::to_string(batch) + ".csv"));
		}
		else if (!m_batch_parallel) {
			std::vector<std::vector<cypress::Real>> spikes;
			auto pop2 = m_networks[batch].populations()[0];
			for (size_t i = 0; i < pop2.size(); i++) {
				spikes.push_back(pop2[i].signals().data(0));
			}
			Utilities::write_vector2_to_csv(
			    spikes, _debug_filename("spikes_input_" +
			                            std::to_string(batch) + ".csv"));
		}

        if (m_count_spikes) {
            for (auto &pop : m_all_pops) {
                std::vector<std::vector<cypress::Real>> spikes;
                for (size_t i = 0; i < pop.size(); i++) {
                    spikes.push_back(pop[i].signals().data(0));
                }
                std::string file = "spikes_pop" + std::to_string(pop.pid()) +
                                   "_batch" + std::to_string(batch) + ".csv";
                Utilities::write_vector2_to_csv(spikes, _debug_filename(file));
                Utilities::plot_spikes(_debug_filename(file), m_backend);
            }
        }
        else {
            std::vector<std::vector<cypress::Real>> spikes;
            for (size_t i = 0; i < pop.size(); i++) {
                spikes.push_back(pop[i].signals().data(0));
            }
            Utilities::write_vector2_to_csv(
                spikes,
                _debug_filename("spikes_" + std::to_string(batch) + ".csv"));
            Utilities::plot_spikes(
                _debug_filename("spikes_" + std::to_string(batch) + ".csv"),
                m_backend);
        }
        /*
         * Currently hard coded
        auto pop2 = m_label_pops[0].network().populations()[1];
        mnist_helper::conv_spikes_per_kernel("testspikes.csv", pop2,
		                                     m_duration, m_pause,
		                                     m_batchsize, 60);
        */
#endif
	}
	if (m_count_spikes) {
		size_t global_count = 0;
		for (auto &pop : m_all_pops) {
			size_t count = 0.0;
			for (auto neuron : pop) {
				count += neuron.signals().data(0).size();
			}
			global_count += count;
			global_logger().info(
			    "SNABSuite", "Pop " + std::to_string(pop.pid()) +
			                     " with size " + std::to_string(pop.size()) +
			                     " fired " + std::to_string(count) + " spikes");
		}
		global_logger().info(
		    "SNABSuite",
		    "Summ of all spikes: " + std::to_string(global_count) + " spikes");
	}
	Real acc = Real(global_correct) / Real(images);
	Real sim_time = m_netw.runtime().sim_pure;
	if (!m_batch_parallel) {
		sim_time = 0.0;
		for (auto &pop : m_label_pops) {
			sim_time += pop.network().runtime().sim_pure;
		}
	}
	if (m_ttfs) {
		std::vector<Real> time_to_sol =
		    TTFS_response_time(m_label_pops, m_batchsize, m_duration, m_pause);
		Real max, min, avg, std_dev;
		Utilities::calculate_statistics(time_to_sol, min, max, avg, std_dev);

		return {std::array<cypress::Real, 4>({acc, NaN(), NaN(), NaN()}),
		        std::array<cypress::Real, 4>({sim_time, NaN(), NaN(), NaN()}),
		        std::array<cypress::Real, 4>({avg, std_dev, min, max})};
	}
	return {std::array<cypress::Real, 4>({acc, NaN(), NaN(), NaN()}),
	        std::array<cypress::Real, 4>({sim_time, NaN(), NaN(), NaN()}),
	        std::array<cypress::Real, 4>({m_duration, NaN(), NaN(), NaN()})};
}

size_t MNIST_BASE::create_deep_network(Network &netw, Real max_weight,
                                       Real max_pool_weight, Real pool_inhib_weight)
{
	size_t layer_id = netw.populations().size();
	if (!m_activity_based_scaling) {
		if (m_weights_scale_factor == 0.0) {
			if (max_weight > 0) {
				m_weights_scale_factor = max_weight / m_mlp->max_weight();
			}
			else {
				m_weights_scale_factor = 1.0;
			}
		}
	}
	else {
		m_weights_scale_factor = max_weight ? max_weight : 1.0;
	}
    if (m_conv_weights_scale_factors.empty()){
		std::string default_conv_name = "conv_max_weight";
		for (size_t i = 0; i < m_mlp->get_conv_layers().size(); i++){
			std::string conv_name = default_conv_name + "_" + std::to_string(i);
			Real layer_max_weight = 0;
			if (!m_config_file[conv_name].empty()){
				layer_max_weight = m_config_file[conv_name].get<Real>();
			} else if (!m_config_file[default_conv_name].empty()){
				layer_max_weight = m_config_file[default_conv_name].get<Real>();
				global_logger().debug("SNABSuite",
			    "Found no conv_max_weight parameter for layer "+std::to_string(i)+".\n"
			    "Number of convolution layers: " + std::to_string(m_mlp->get_conv_layers().size())+
				". Falling back on default conv_max_weight value.");
			} else {
				global_logger().debug("SNABSuite",
			    "Found no conv_max_weight parameter for layer "+std::to_string(i)+
				" and no default conv_max_weight parameter.\n"
			    "Number of convolution layers: " + std::to_string(m_mlp->get_conv_layers().size())+".");
			}
			if (layer_max_weight > 0){
                m_conv_weights_scale_factors.push_back(
                        layer_max_weight / m_mlp->conv_max_weight(i));
			} else {
			    m_conv_weights_scale_factors.push_back(1.0);
            }
		}
	}

	size_t dense_counter = 0;
	size_t conv_counter = 0;
	size_t pool_counter = 0;
	for (const auto &layer : m_mlp->get_layer_types()){
        if (layer == mnist_helper::Dense){
			const auto &layer_weights = m_mlp->get_weights()[dense_counter];
            size_t size = layer_weights.cols();
            auto pop = SpikingUtils::add_population(m_neuron_type_str, netw,
                                                    m_neuro_params, size, "");
            auto conns = mnist_helper::dense_weights_to_conn(
                layer_weights, m_weights_scale_factor, m_syn_delay);
            netw.add_connection(netw.populations()[layer_id - 1], pop,
                                Connector::from_list(conns),
                                ("dense_" + std::to_string(dense_counter)).c_str());

            global_logger().debug(
                "SNABSuite",
                "Dense layer constructed with size " + std::to_string(size));

			dense_counter++;
		} else if (layer == mnist_helper::Conv){
			const auto &layer_weights = m_mlp->get_conv_layers()[conv_counter];
            size_t size = layer_weights.output_sizes[0] * layer_weights.output_sizes[1] * layer_weights.output_sizes[2];
			auto pop = SpikingUtils::add_population(m_neuron_type_str, netw,
			                                        m_neuro_params, size, "");
			auto conns = mnist_helper::conv_weights_to_conn(
			    layer_weights, m_conv_weights_scale_factors[conv_counter], 1.0);
			netw.add_connection(netw.populations()[layer_id - 1], pop,
			                    Connector::from_list(conns),
                                ("conv_" + std::to_string(conv_counter)).c_str());
            global_logger().debug(
			    "SNABSuite",
			    "Convolution layer constructed with size " + std::to_string(size));
			conv_counter++;
		} else if (layer == mnist_helper::Pooling){
			auto &pool_layer = m_mlp->get_pooling_layers()[pool_counter];
			size_t size = pool_layer.output_sizes[0] * pool_layer.output_sizes[1]
			                * pool_layer.output_sizes[2];
			auto pop = SpikingUtils::add_population(m_neuron_type_str, netw,
			                                        m_neuro_params, size, "");
			auto conns = mnist_helper::pool_to_conn(pool_layer, max_pool_weight,
			                                        pool_inhib_weight, 1.0, m_pool_delay);
			netw.add_connection(netw.populations()[layer_id - 1],
			                    netw.populations()[layer_id - 1],
			                    Connector::from_list(conns[0]),
			                    "dummy_name");
			netw.add_connection(netw.populations()[layer_id - 1], pop,
			                    Connector::from_list(conns[1]),
                                ("pool_" + std::to_string(pool_counter)).c_str());
			global_logger().debug(
			    "SNABSuite",
			    "Pooling layer constructed with size " + std::to_string(size) +
			    " and " + std::to_string(conns[0].size()) + " inhibitory connections");
			pool_counter++;
		}
        layer_id++;
	}
	return dense_counter+conv_counter+pool_counter;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cypress::Network &MnistITLLastLayer::build_netw(cypress::Network &netw)
{
	read_config();

	if (m_config_file.find("positive") != m_config_file.end()) {
		m_positive = m_config_file["positive"].get<bool>();
	}
	if (m_config_file.find("norm_rate_hidden") != m_config_file.end()) {
		m_norm_rate_hidden = m_config_file["norm_rate_hidden"].get<Real>();
	}
	if (m_config_file.find("norm_rate_last") != m_config_file.end()) {
		m_norm_rate_last = m_config_file["norm_rate_last"].get<Real>();
	}
	if (m_config_file.find("loss_function") != m_config_file.end()) {
		m_loss_function = m_config_file["loss_function"].get<std::string>();
	}
	bool random_init = false;
	if (m_config_file.find("random_init") != m_config_file.end()) {
		random_init = m_config_file["random_init"].get<bool>();
	}
	if (m_config_file.find("num_test_images") != m_config_file.end()) {
		m_num_test_images = m_config_file["num_test_images"].get<size_t>();
	}
	if (m_config_file.find("test_batchsize") != m_config_file.end()) {
		m_test_batchsize = m_config_file["test_batchsize"].get<size_t>();
	}
	else {
		m_test_batchsize = m_batchsize;
	}

	// TODO ? Required -> constructor

	auto kerasdata = mnist_helper::read_network(m_dnn_file, true);
	if (m_positive) {
		if (m_loss_function == "CatHinge")
			m_mlp = std::make_shared<MNIST::MLP<MNIST::CatHinge, MNIST::ReLU,
			                                    MNIST::PositiveLimitedWeights>>(
			    kerasdata, m_config_file["epochs"].get<size_t>(), m_batchsize,
			    m_config_file["learn_rate"].get<Real>(), random_init);
		else if (m_loss_function == "MSE")
			m_mlp = std::make_shared<MNIST::MLP<MNIST::MSE, MNIST::ReLU,
			                                    MNIST::PositiveLimitedWeights>>(
			    kerasdata, m_config_file["epochs"].get<size_t>(), m_batchsize,
			    m_config_file["learn_rate"].get<Real>(), random_init);
		else {
			throw std::runtime_error("Unknown loss function " +
			                         m_loss_function);
		}
	}
	else {
		if (m_loss_function == "CatHinge")
			m_mlp = std::make_shared<
			    MNIST::MLP<MNIST::CatHinge, MNIST::ReLU, MNIST::NoConstraint>>(
			    kerasdata, m_config_file["epochs"].get<size_t>(), m_batchsize,
			    m_config_file["learn_rate"].get<Real>(), random_init);
		else if (m_loss_function == "MSE")
			m_mlp = std::make_shared<
			    MNIST::MLP<MNIST::MSE, MNIST::ReLU, MNIST::NoConstraint>>(
			    kerasdata, m_config_file["epochs"].get<size_t>(), m_batchsize,
			    m_config_file["learn_rate"].get<Real>(), random_init);
		else {
			throw std::runtime_error("Unknown loss function " +
			                         m_loss_function);
		}
	}

	if (m_scaled_image) {
		m_mlp->scale_down_images();
	}
	m_spmnist =
	    mnist_helper::mnist_to_spike(m_mlp->mnist_train_set(), m_duration,
	                                 m_max_freq, m_images, m_poisson, m_ttfs);
	if (m_activity_based_scaling) {
		m_layer_scale_factors =
		    m_mlp->rescale_weights(m_activity_based_scaling);  // TODO
		std::string message;
		for (auto i : m_layer_scale_factors) {
			message += std::to_string(i);
			message += ", ";
		}
		global_logger().debug("SNABSuite", "SNN rescale factors: " + message);
	}
	return netw;
}

void MnistITLLastLayer::run_netw(cypress::Network &netw)
{
	cypress::PowerManagementBackend pwbackend(
	    cypress::Network::make_backend(m_backend));

	auto source_n = netw.create_population<SpikeSourceArray>(
	    m_mlp->get_layer_sizes()[0], SpikeSourceArrayParameters(),
	    SpikeSourceArraySignals(), "input_layer");

	create_deep_network(netw, m_max_weight,
	                    m_max_pool_weight, m_pool_inhib_weight);
	m_label_pops = {netw.populations().back()};

	auto pre_last_pop = netw.populations()[netw.populations().size() - 2];
	if (m_last_layer_only && !m_count_spikes) {
		m_label_pops[0].signals().record(0);
		pre_last_pop.signals().record(0);
	}
	else {
		m_all_pops.clear();
		for (auto pop : netw.populations()) {
			pop.signals().record(0);
			m_all_pops.emplace_back(pop);
		}
	}

	std::vector<std::vector<Real>> accuracies;
	std::vector<std::vector<std::vector<Real>>> weight_changes;
	for(size_t layer=0; layer<m_mlp->get_weights().size(); layer++)
	{
		weight_changes.emplace_back(std::vector<std::vector<Real>>());
	}

	size_t counter = 0;
	for (size_t train_run = 0; train_run < m_config_file["epochs"];
	     train_run++) {
		global_logger().debug("MNIST", "next epoch\n");
		m_batch_data = mnist_helper::create_batches(m_spmnist, m_batchsize,
		                                            m_duration, m_pause, true);
		int batch_count=0;
		for (auto &i : m_batch_data) {
			batch_count++;
			global_logger().info("MNIST", "Batch "+std::to_string(batch_count)+"/"+std::to_string(m_batch_data.size()));
			if (std::get<1>(i).size() != m_batchsize) {
				continue;
			}
			mnist_helper::update_spike_source(source_n, i);
			netw.run(pwbackend, m_batchsize * (m_duration + m_pause));

			if (m_ttfs) {
				// TODO adapt for ttfs
				/*Plan:
				 * mit S4NN vergleichen
				 * E-Mail-Feedback
				 * Unit-Test
				 * ggf. weight regularization
				 * bessere Plots
				 * */
				std::vector<cypress::Matrix<Real>> oldW = m_mlp->get_weights();

				backward_path_TTFS(std::get<1>(i), m_mlp->get_weights(), netw.populations(), m_last_layer_only);
//				mnist_helper::update_conns_from_mat(
//					m_mlp->get_weights(), netw, 1.0, m_weights_scale_factor);
				mnist_helper::update_conns_from_mat(
				    m_mlp->get_weights(), netw, 1.0, m_weights_scale_factor);

				std::vector<cypress::Matrix<Real>> newW = m_mlp->get_weights();

				for(size_t layer=0; layer<oldW.size(); layer++)
				{
					cypress::Real absSum=0;
					cypress::Real oldMin=oldW[0](0,0);
					cypress::Real oldMax=oldW[0](0,0);
					cypress::Real newMin=newW[0](0,0);
					cypress::Real newMax=newW[0](0,0);
					for(size_t from=0; from < oldW[layer].rows(); from++)
					{
						for(size_t to=0; to<oldW[layer].cols(); to++)
						{
							absSum+=abs(oldW[layer](from, to)-newW[layer](from, to));
							if(oldW[layer](from, to)<oldMin)
							{
								oldMin=oldW[layer](from, to);
							}
							if(oldW[layer](from, to)>oldMax)
							{
								oldMax=oldW[layer](from, to);
							}
							if(newW[layer](from, to)<newMin)
							{
								newMin=newW[layer](from, to);
							}
							if(newW[layer](from, to)>newMax)
							{
								newMax=newW[layer](from, to);
							}
						}
					}
				    absSum/= oldW[layer].cols()*oldW[layer].rows();
					weight_changes[layer].emplace_back(std::vector<Real>{Real(counter) / Real(m_batch_data.size()), absSum});
					global_logger().debug("MNIST", "Weights of layer "+std::to_string(layer)+" changed from "+std::to_string(oldMin)+" - "+
						std::to_string(oldMax)+" to "+std::to_string(newMin)+" -"+std::to_string(newMax)+" with differences of "+std::to_string(absSum));
				}
			}
			else {
				// BEGIN OF RATE_CODING
				std::vector<std::vector<std::vector<Real>>> output_rates;
				for (auto &pop : netw.populations()) {
                    //append for each layer the output rates
					if (pop.signals().is_recording(0)) {
						// if (!m_ttfs)
						if (pop.pid() != netw.populations().back().pid()) {
							output_rates.emplace_back(
							    mnist_helper::spikes_to_rates(
							        pop, m_duration, m_pause, m_batchsize,
							        m_norm_rate_hidden));
						}
						else {
							output_rates.emplace_back(
							    mnist_helper::spikes_to_rates(
							        pop, m_duration, m_pause, m_batchsize,
							        m_norm_rate_last));
						}
						/*}
						else {
						    output_rates.emplace_back(
						        mnist_helper::spikes_to_rates_ttfs(
						            pop, m_duration, m_pause, m_batchsize));
						}*/
					}
					else {
						output_rates.emplace_back(
						    std::vector<std::vector<Real>>());
					}
				}
				//backwards path
				m_mlp->backward_path_2(std::get<1>(i), output_rates,
				                       m_last_layer_only);

				mnist_helper::update_conns_from_mat(
				    m_mlp->get_weights(), netw, 1.0, m_weights_scale_factor);
				// END OF RATE_CODING
			}

			// Calculate batch accuracy
			auto labels = mnist_helper::spikes_to_labels(
			    m_label_pops[0], m_duration, m_pause, m_batchsize, m_ttfs);
			m_global_correct =
			    mnist_helper::compare_labels(std::get<1>(i), labels);
			m_num_images = std::get<1>(i).size();
			m_sim_time = netw.runtime().sim_pure;
			global_logger().debug(
			    "SNABsuite",
			    "Batch accuracy: " + std::to_string(Real(m_global_correct) /
			                                        Real(m_num_images)));

			accuracies.emplace_back(
			    std::vector<Real>{Real(counter) / Real(m_batch_data.size()),
			                      Real(m_global_correct) / Real(m_num_images)});
			counter++;
		}
	}

	m_global_correct = 0;
	m_num_images = 0;
	m_sim_time = 0.0;
	size_t global_count = 0;
	std::vector<size_t> local_count, pop_size, pids;
	mnist_helper::SPIKING_MNIST test_data;
	if (m_train_data == false) {
		test_data = mnist_helper::mnist_to_spike(
		    m_mlp->mnist_test_set(), m_duration, m_max_freq, m_num_test_images,
		    m_poisson, m_ttfs);
	}
	else {
		test_data = mnist_helper::mnist_to_spike(
		    m_mlp->mnist_train_set(), m_duration, m_max_freq, m_num_test_images,
		    m_poisson, m_ttfs);
	}
	m_batch_data = mnist_helper::create_batches(test_data, m_test_batchsize,
	                                            m_duration, m_pause, true);
	for (auto &i : m_batch_data) {
		mnist_helper::update_spike_source(source_n, i);
		netw.run(pwbackend, m_test_batchsize * (m_duration + m_pause));

		auto pop = m_label_pops[0];
		auto labels = mnist_helper::spikes_to_labels(pop, m_duration, m_pause,
		                                             m_test_batchsize, m_ttfs);
		auto &orig_labels = std::get<1>(i);
		auto correct = mnist_helper::compare_labels(orig_labels, labels);
		m_global_correct += correct;
		m_num_images += orig_labels.size();
		m_sim_time += netw.runtime().sim_pure;
		if (m_count_spikes) {
			for (auto &pop : m_all_pops) {
				size_t count = 0.0;
				for (auto neuron : pop) {
					count += neuron.signals().data(0).size();
				}
				global_count += count;
				local_count.emplace_back(count);
				pop_size.emplace_back(pop.size());
				pids.emplace_back(pop.pid());
			}
		}
	}

	if (m_count_spikes) {
		for (size_t i = 0; i < local_count.size(); i++) {
			global_logger().info(
			    "SNABSuite", "Pop " + std::to_string(pids[i]) + " with size " +
			                     std::to_string(pop_size[i]) + " fired " +
			                     std::to_string(local_count[i]) + " spikes");
		}
		global_logger().info(
		    "SNABSuite",
		    "Summ of all spikes: " + std::to_string(global_count) + " spikes");
	}

#if SNAB_DEBUG
	std::vector<std::vector<cypress::Real>> spikes_pre;
	for (size_t i = 0; i < pre_last_pop.size(); i++) {
		spikes_pre.push_back(pre_last_pop[i].signals().data(0));
	}
	Utilities::write_vector2_to_csv(spikes_pre,
	                                _debug_filename("spikes_pre.csv"));
	Utilities::plot_spikes(_debug_filename("spikes_pre.csv"), m_backend);

	spikes_pre.clear();
	for (size_t i = 0; i < m_label_pops[0].size(); i++) {
		spikes_pre.push_back(m_label_pops[0][i].signals().data(0));
	}
	Utilities::write_vector2_to_csv(spikes_pre,
	                                _debug_filename("spikes_label.csv"));
	Utilities::plot_spikes(_debug_filename("spikes_label.csv"), m_backend);
#endif

	Utilities::write_vector2_to_csv(accuracies,
	                                _debug_filename("accuracies.csv"));
	Utilities::plot_1d_curve(_debug_filename("accuracies.csv"), m_backend, 0,
	                         1);

	//Plots averaged weight updates for each layer,
	// needs the calculation of those averaged updates below the call of backward_path_TTFS
	/*for(size_t layer=0; layer<weight_changes.size(); layer++)
	{
		Utilities::write_vector2_to_csv(weight_changes[layer],
			                            _debug_filename("weightChanges"+std::to_string(layer)+".csv"));
		Utilities::plot_1d_curve(_debug_filename("weightChanges"+std::to_string(layer)+".csv"), m_backend, 0,
			                     1);
	}*/
}

std::vector<std::array<cypress::Real, 4>> MnistITLLastLayer::evaluate()
{
	Real acc = Real(m_global_correct) / Real(m_num_images);
	return {std::array<cypress::Real, 4>({acc, NaN(), NaN(), NaN()}),
	        std::array<cypress::Real, 4>({m_sim_time, NaN(), NaN(), NaN()})};
}

Matrix<Real> MnistITLLastLayer::compute_backprop_mat(const std::vector<PopulationBase> structure, const std::vector<std::vector<cypress::Real>>& spike_times,Matrix<Real> weights, const int layer) //target-layer // Copy Weights here!
{
	#ifndef NDEBUG
		assert(layer>=1);
	#endif
	for(size_t i=0; i<structure[layer].size(); i++)
	{
		for(size_t j=0; j<structure[layer+1].size(); j++)
		{
            if(!(spike_times[layer][i]+m_syn_delay/(m_duration+m_pause)<spike_times[layer+1][j]))
            {
                weights(i,j)= 0.0;
            }
		}
	}
	return weights;
}

namespace{
cypress::Real norm(const std::vector<cypress::Real>& vec){
    Real sum = 0;
    for(auto& i: vec)
        sum += i*i;
    if (sum==0)
        return 1;
    return std::sqrt(sum);
}
void normalize(std::vector<cypress::Real> &vec){
    auto n = norm(vec);
    for(auto& i: vec){
        i/=n;
    }
}
}

void MnistITLLastLayer::backward_path_TTFS(
	const std::vector<uint16_t> &labels, std::vector<cypress::Matrix<Real>> &weights,
    std::vector<PopulationBase> populations, bool last_only=false)
{
	const std::vector<cypress::Matrix<cypress::Real>> orig_weights = weights;

	std::vector<std::vector<std::vector<cypress::Real>>> spike_times=mnist_helper::getSpikeTimes(populations, m_duration, m_pause, m_batchsize);


	//bool noPlotYet=true;
	for (size_t sample = 0; sample < m_batchsize; sample++) {

		// debug plots
		// Shows the first spikes of the output layer for the first image labeled X in each batch
        /*std::cout << labels[sample] <<std::endl;
		if(labels[sample]==5 && noPlotYet && batch%10==1)
		{
			noPlotYet=false;
			std::vector<std::vector<cypress::Real>> plotSpikeTimes;
			for(size_t neuronIndex = 0; neuronIndex<populations[populations.size()-1].size(); neuronIndex++)
			{
				cypress::Real spikeTime = spike_times[sample][populations.size()-1][neuronIndex];
				plotSpikeTimes.emplace_back(std::vector<cypress::Real>{spikeTime});
			}
			plotSpikeTimes.emplace_back(std::vector<cypress::Real>{1}); //last artificial neuron to force equally sized plots over batches

			Utilities::write_vector2_to_csv(plotSpikeTimes,
			                                _debug_filename("spike_times"+std::to_string(batch)+".csv"));
			Utilities::plot_spikes(_debug_filename("spike_times"+std::to_string(batch)+".csv"), m_backend);
		}*/
		std::vector<cypress::Real> errors;
		for(size_t layer = populations.size()-1; layer>0; layer--)
		{

            /*std::vector<cypress::Real> errors;std::cout << "spike_times"<<std::endl;
            for(auto i : spike_times[sample][layer]){
                std::cout << i <<",";
            }
            std::cout <<std::endl;*/

			if(layer==populations.size()-1)
			{
				errors=compute_TTFS_error(labels[sample], spike_times[sample][populations.size()-1]);
			}
			else{
                auto mat =  compute_backprop_mat(populations, spike_times[sample], orig_weights[layer], layer);
				errors = mat_X_vec(mat, errors);
                normalize(errors);
			}


			Matrix<Real> gradients = compute_gradients(populations, spike_times[sample], errors, layer);
			update_mat_TTFS(weights[layer-1], gradients, m_batchsize,
			                m_mlp->learnrate());
		}


		//TODO m_constraint.constrain_weights(m_layers);
	}
	//TODO m_scaled_layerwise = false;
}
/**
 *
 * @param label ground truth
 * @param spike_times First spike times of all neurons in the last layer.
 * @return
 */
std::vector<cypress::Real> MnistITLLastLayer::compute_TTFS_error(const uint16_t label, const std::vector<cypress::Real> &spike_times)
{
	std::vector<cypress::Real> errors;
	cypress::Real min_time=*min_element(spike_times.begin(), spike_times.end());

	int expected_later_counter=0;
	int okay_counter=0;

	for (size_t i = 0; i < spike_times.size(); i++) {
		if(min_time==1)
		{
			errors.push_back(i==label ? -m_gamma/(m_pause+m_duration)  : 0);
		}
		else
		{
			Real expected_time;
			if(i==label)
			{
				expected_time=min_time;
			}
			else if (spike_times[i]-min_time < m_gamma/(m_pause+m_duration))
			{
				expected_time=std::min(min_time+m_gamma/(m_pause+m_duration), cypress::Real(1));
				expected_later_counter++;
			}
			else
			{
				expected_time=spike_times[i];
				okay_counter++;
			}
			errors.push_back(expected_time-spike_times[i]); // divided by t_max, but t_max is 1
		}
	}
    normalize(errors);
	return errors;
}



/**
 *
 * @param spike_times
 * @param deltas
 * @param i
 * @param j
 * @param layer target layer. in [1, #layers]
 * @return
 */
cypress::Real MnistITLLastLayer::compute_TTFS_gradient(const std::vector<std::vector<cypress::Real>> &spike_times, const std::vector<cypress::Real> &errors, const int i, const int j, const int layer) //dL/dw_ji^l
{
	if(spike_times[layer-1][i]+m_syn_delay/(m_duration+m_pause)<spike_times[layer][j])
	{
		return errors[j];
	}
	else {
		return 0;
	}
}

Matrix<Real> MnistITLLastLayer::compute_gradients(const std::vector<PopulationBase> structure, const std::vector<std::vector<cypress::Real>> spike_times, const std::vector<cypress::Real> errors, const int layer) //target-layer
{
	#ifndef NDEBUG
		assert(layer>=1);
	#endif
	Matrix<Real> gradients(structure[layer-1].size(), structure[layer].size());
	for(size_t i=0; i<structure[layer-1].size(); i++)
	{
		for(size_t j=0; j<structure[layer].size(); j++)
		{
			gradients(i,j)= compute_TTFS_gradient(spike_times, errors, i, j, layer);
		}
	}
	return gradients;
}



void MnistITLLastLayer::update_mat_TTFS(Matrix<Real> &mat, const Matrix<Real> &gradients, const size_t sample_num, const Real learn_rate)
{
	Real sample_num_r(sample_num);

	for (size_t i = 0; i < mat.rows(); i++) {
		for (size_t j = 0; j < mat.cols(); j++) {
			mat(i, j) = mat(i, j) - learn_rate * gradients(i,j) / sample_num_r;
           //mat(i, j) = mat(i, j) - learn_rate * 0.01 * mat(i, j);
		}
	}
}

std::vector<Real> MnistITLLastLayer::mat_X_vec(const cypress::Matrix<cypress::Real> &mat,
                                          const std::vector<cypress::Real> &vec)
{
#ifndef NDEBUG
	assert(mat.cols() == vec.size());
#endif
	std::vector<Real> res(mat.rows(), 0.0);
	for (size_t i = 0; i < mat.rows(); i++) {
		for (size_t j = 0; j < mat.cols(); j++) {
			res[i] += mat(i, j) * vec[j];
		}
	}
	return res;
}


}  // namespace SNAB
