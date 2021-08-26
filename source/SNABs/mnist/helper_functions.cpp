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
#include <assert.h>

#include <chrono>
#include <cmath>
#include <cypress/cypress.hpp>
#include <fstream>
#include <string>
#include <utility>  //std::pair

#include "helper_functions.hpp"

namespace mnist_helper {
using namespace cypress;

MNIST_DATA loadMnistData(const size_t num_data, const std::string path)
{
	MNIST_DATA res;
	std::ifstream images, labels;
	images.open(path + "-images-idx3-ubyte", std::ios::binary);
	if (!images.good()) {
		throw std::runtime_error("Could not open image file " + path +
		                         "-images-idx3-ubyte!");
	}
	labels.open(path + "-labels-idx1-ubyte", std::ios::binary);
	if (!images.good()) {
		throw std::runtime_error("Could not open label file " + path +
		                         "-labels-idx1-ubyte");
	}

	images.seekg(16, images.beg);
	unsigned char tmp = 0;
	for (size_t i = 0; i < num_data; i++) {
		std::vector<Real> image;
		for (size_t row = 0; row < 28; row++) {
			for (size_t col = 0; col < 28; col++) {
				images.read((char *)&tmp, sizeof(tmp));
				if (images.eof()) {
					throw std::runtime_error("Error reading file!");
				}
				image.push_back(((Real)tmp) / 255.0);
			}
		}
		std::get<0>(res).push_back(image);
	}

	labels.seekg(8, labels.beg);
	for (size_t i = 0; i < num_data; i++) {
		labels.read((char *)&tmp, sizeof(tmp));
		std::get<1>(res).push_back(uint16_t(tmp));
	}

	images.close();
	labels.close();

	return res;
}
void print_image(const std::vector<Real> &img, size_t wrap)
{
	size_t count = 0;
	for (auto i : img) {
		if (i > 0.5) {
			std::cout << "#";
		}
		else {
			std::cout << " ";
		}
		count++;
		if (count == wrap) {
			std::cout << std::endl;
			count = 0;
		}
	}
}

std::vector<std::vector<std::vector<Real>>> image_to_rate(
    const std::vector<std::vector<Real>> &images, const Real duration,
    const Real max_freq, size_t num_images, bool poisson)
{
	std::vector<std::vector<std::vector<Real>>> rate_images;
	for (size_t i = 0; i < num_images; i++) {
		std::vector<std::vector<Real>> spike_image;
		for (const auto &pixel : images[i]) {
			if (poisson) {
				spike_image.emplace_back(
				    cypress::spikes::poisson(0.0, duration, max_freq * pixel));
			}
			else {
				spike_image.emplace_back(cypress::spikes::constant_frequency(
				    0.0, duration, max_freq * pixel));
			}
		}
		rate_images.emplace_back(spike_image);
	}
	return rate_images;
}

std::vector<std::vector<std::vector<Real>>> image_to_TTFS(
    const std::vector<std::vector<Real>> &images, const Real duration,
    size_t num_images)
{
	std::vector<std::vector<std::vector<Real>>> TTFS_images;
	for (size_t i = 0; i < num_images; i++) {
		std::vector<std::vector<Real>> spike_image;
		for (const auto &pixel : images[i]) {
			if (pixel > 0) {
				spike_image.emplace_back(
				    std::vector<Real>{(1.0 - pixel) * duration});
			}
			else {
				spike_image.emplace_back(std::vector<Real>{});
			}
		}
		TTFS_images.emplace_back(spike_image);
	}
	return TTFS_images;
}

SPIKING_MNIST mnist_to_spike(const MNIST_DATA &mnist_data, const Real duration,
                             const Real max_freq, size_t num_images,
                             bool poisson, bool ttfs)
{
	SPIKING_MNIST res;
	if (!ttfs) {
		std::get<0>(res) = image_to_rate(std::get<0>(mnist_data), duration,
		                                 max_freq, num_images, poisson);
	}
	else {
		std::get<0>(res) =
		    image_to_TTFS(std::get<0>(mnist_data), duration, num_images);
	}
	std::get<1>(res) = std::get<1>(mnist_data);
	return res;
}

std::vector<MNIST_DATA> create_batches(const SPIKING_MNIST &mnist_data,
                                       const size_t batch_size, Real duration,
                                       Real pause, const bool shuffle,
                                       unsigned seed)
{
	std::vector<size_t> indices(std::get<0>(mnist_data).size());
	for (size_t i = 0; i < indices.size(); i++) {
		indices[i] = i;
	}
	if (shuffle) {
		if (seed == 0) {
			seed = std::chrono::system_clock::now().time_since_epoch().count();
		}
		auto rng = std::default_random_engine{seed};
		std::shuffle(indices.begin(), indices.end(), rng);
	}
	size_t counter = 0;
	size_t image_size = std::get<0>(mnist_data)[0].size();
	std::vector<MNIST_DATA> res;
	while (counter < indices.size()) {
		MNIST_DATA single_batch_combined;

		std::vector<std::vector<Real>> &single_batch =
		    std::get<0>(single_batch_combined);
		for (size_t pixel = 0; pixel < image_size; pixel++) {
			std::vector<Real> spike_pxl;
			for (size_t index = 0; index < batch_size; index++) {
				if (counter + index >= indices.size()) {
					break;
				}
				size_t shfld_index = indices[counter + index];

				auto pxl_spk = std::get<0>(mnist_data)[shfld_index][pixel];
				std::for_each(pxl_spk.begin(), pxl_spk.end(),
				              [&duration, &pause, &index](Real &d) {
					              d += (duration + pause) * index;
				              });
				spike_pxl.insert(spike_pxl.end(), pxl_spk.begin(),
				                 pxl_spk.end());
			}
			single_batch.emplace_back(spike_pxl);
		}
		std::vector<uint16_t> &labels = std::get<1>(single_batch_combined);
		for (size_t index = 0; index < batch_size; index++) {
			if (counter + index >= indices.size()) {
				break;
			}
			size_t shfld_index = indices[counter + index];
			labels.emplace_back(std::get<1>(mnist_data)[shfld_index]);
		}
		res.push_back(std::move(single_batch_combined));
		counter += batch_size;
	}
	return res;
}

cypress::Population<SpikeSourceArray> create_spike_source(
    Network &netw, const MNIST_DATA &spikes)
{
	size_t size = std::get<0>(spikes).size();

	auto pop = netw.create_population<SpikeSourceArray>(
	    size, SpikeSourceArrayParameters(), SpikeSourceArraySignals(),
	    "input_layer");
	for (size_t nid = 0; nid < size; nid++) {
		pop[nid].parameters().spike_times(std::get<0>(spikes)[nid]);
	}
	return pop;
}

cypress::Population<SpikeSourceArray> &update_spike_source(
    cypress::Population<SpikeSourceArray> &source, const MNIST_DATA &spikes)
{
	size_t size = std::get<0>(spikes).size();

	if (source.size() != size) {
		throw std::runtime_error(
		    "Spike source array size does not equal image size!");
	}
	for (size_t nid = 0; nid < size; nid++) {
		source[nid].parameters().spike_times(std::get<0>(spikes)[nid]);
	}
	return source;
}

Json read_network(std::string path, bool msgpack)
{
	std::ifstream file_in;
	Json json;
	if (msgpack) {
		file_in.open(path, std::ios::binary);
		if (!file_in.good()) {
			file_in.open("../" + path, std::ios::binary);
			if (!file_in.good()) {
				throw std::runtime_error("Could not open deep network file " +
				                         path);
			}
		}
		json = Json::from_msgpack(file_in);
	}
	else {
		file_in.open(path, std::ios::binary);
		if (!file_in.good()) {

			file_in.open("../" + path, std::ios::binary);
			if (!file_in.good()) {
				throw std::runtime_error("Could not open deep network file " +
				                         path);
			}
		}
		json = Json::parse(file_in);
	}
	return json;
}

std::vector<LocalConnection> dense_weights_to_conn(const Matrix<Real> &mat,
                                                   Real scale, Real delay)
{
	std::vector<LocalConnection> conns;
	for (size_t i = 0; i < mat.rows(); i++) {
		for (size_t j = 0; j < mat.cols(); j++) {
			Real w = mat(i, j);
			conns.emplace_back(LocalConnection(i, j, scale * w, delay));
		}
	}
	return conns;
}

std::vector<LocalConnection> conv_weights_to_conn(
    const mnist_helper::CONVOLUTION_LAYER &layer,
    Real scale, Real delay)
{
    std::vector<LocalConnection> conns;
	size_t stride = layer.stride;
	size_t kernel_size_x = layer.filter.size();
	size_t kernel_size_y = layer.filter[0].size();
	size_t kernel_size_z = layer.filter[0][0].size();
	size_t max_x = layer.input_sizes[0] - kernel_size_x + 1;
    size_t max_y = layer.input_sizes[1] - kernel_size_y + 1;

    for (size_t i = 0; i < max_x; i += stride) {
        for (size_t j = 0; j < max_y; j += stride) {
            for (size_t filter = 0; filter < layer.output_sizes[2]; filter++){
                for (size_t x = 0; x < kernel_size_x; x++) {
                    for (size_t y = 0; y < kernel_size_y; y++) {
                        for (size_t z = 0; z < kernel_size_z; z++) {
                            conns.emplace_back((LocalConnection(
                                (i + x) * layer.input_sizes[1] * layer.input_sizes[2] +
                                (j + y) * layer.input_sizes[2] +
                                z,
                                i * layer.output_sizes[2] * layer.output_sizes[1] +
                                j * layer.output_sizes[2] +
                                filter,
                                scale * layer.filter[x][y][z][filter], delay)));
						}
                    }
                }
            }
        }
    }
    return conns;
}

std::vector<std::vector<LocalConnection>> pool_to_conn(
    const mnist_helper::POOLING_LAYER &layer, Real max_pool_weight,
    Real pool_inhib_weight, Real delay, Real pool_delay)
{
    std::vector<LocalConnection> inhib_conns;
	std::vector<LocalConnection> pool_cons;
	std::vector<std::vector<LocalConnection>> conns;
    size_t max_x = layer.input_sizes[0]-ceil(layer.size[0]/2);
	size_t max_y = layer.input_sizes[1]-ceil(layer.size[1]/2);
	for (size_t i = 0; i < max_x; i += layer.stride){
		for (size_t j = 0; j < max_y; j += layer.stride){
			for (size_t k = 0; k < layer.input_sizes[2]; k++){
				for (size_t x = 0; x < layer.size[0]; x++){
					for (size_t y = 0; y < layer.size[1]; y++){
                        for (size_t u = 0; u < layer.size[0]; u++){
                            for (size_t v = 0; v < layer.size[1]; v++){
								if (u != x || v != y) {
									inhib_conns.emplace_back(LocalConnection(
                                        (i + x) * layer.input_sizes[1] * layer.input_sizes[2] +
									    (j + y) * layer.input_sizes[2] +
									    k,
									    (i + u) * layer.input_sizes[1] * layer.input_sizes[2] +
									    (j + v) * layer.input_sizes[2] +
									    k,
									    pool_inhib_weight, pool_delay));
								}
                            }
                        }
						pool_cons.emplace_back(LocalConnection(
                            (i + x) * layer.input_sizes[1] * layer.input_sizes[2] +
                            (j + y) * layer.input_sizes[2] +
                            k,
                            i/layer.stride * layer.output_sizes[1] * layer.output_sizes[2] +
                            j/layer.stride * layer.output_sizes[2] +
                            k,
                            max_pool_weight, delay));
					}
				}
			}
		}
	}
    conns.push_back(inhib_conns);
    conns.push_back(pool_cons);
	return conns;
}

std::vector<uint16_t> spikes_to_labels(const PopulationBase &pop, Real duration,
                                       Real pause, size_t batch_size, bool ttfs)
{
	std::vector<uint16_t> res(batch_size);
	if (!ttfs) {
		std::vector<std::vector<uint16_t>> binned_spike_counts;
		for (const auto &neuron : pop) {
			binned_spike_counts.push_back(
			    SpikingUtils::spike_time_binning<uint16_t>(
			        -pause * 0.5,
			        batch_size * (duration + pause) - (pause * 0.5), batch_size,
			        neuron.signals().data(0)));
		}

		for (size_t sample = 0; sample < batch_size; sample++) {
			uint16_t max = 0;
			uint16_t index = std::numeric_limits<uint16_t>::max();
			for (size_t neuron = 0; neuron < binned_spike_counts.size();
			     neuron++) {
				if (binned_spike_counts[neuron][sample] > max) {
					index = neuron;
					max = binned_spike_counts[neuron][sample];
				}
				else if (binned_spike_counts[neuron][sample] == max) {
					// Multiple neurons have the same decision
					index = std::numeric_limits<uint16_t>::max();
				}
			}
			res[sample] = index;
		}
	}
	else {
		std::vector<std::vector<Real>> binned_spike_times;
		for (const auto &neuron : pop) {
			binned_spike_times.push_back(SpikingUtils::spike_time_binning_TTFS(
			    0.0, Real(batch_size) * (duration + pause), batch_size,
			    neuron.signals().data(0)));
		}

		for (size_t sample = 0; sample < batch_size; sample++) {
			Real min = std::numeric_limits<Real>::max();
			uint16_t index = std::numeric_limits<uint16_t>::max();
			for (size_t neuron = 0; neuron < binned_spike_times.size();
			     neuron++) {
				if (binned_spike_times[neuron][sample] < min) {
					index = neuron;
					min = binned_spike_times[neuron][sample];
				}
				else if (binned_spike_times[neuron][sample] == min) {
					// Multiple neurons have the same decision
					index = std::numeric_limits<uint16_t>::max();
				}
			}
			res[sample] = index;
		}
	}
	return res;
}

std::vector<std::vector<Real>> spikes_to_rates_ttfs(const PopulationBase pop,
                                                    Real duration, Real pause,
                                                    size_t batch_size)
{
	std::vector<std::vector<Real>> binned_spike_times;
	std::vector<std::vector<Real>> res(batch_size,
	                                   std::vector<Real>(pop.size()));

	Real new_duration = duration;
	for (const auto &neuron : pop) {
		binned_spike_times.push_back(SpikingUtils::spike_time_binning_TTFS(
		    0.0, Real(batch_size) * (duration + pause), batch_size,
		    neuron.signals().data(0)));
	}
	for (size_t sample = 0; sample < batch_size; sample++) {
		Real min = std::numeric_limits<Real>::max();
		// Real max = 0.0;
		for (size_t neuron = 0; neuron < binned_spike_times.size(); neuron++) {
			if (binned_spike_times[neuron][sample] < min) {
				min = binned_spike_times[neuron][sample];
			}
			/*if (binned_spike_times[neuron][sample] > max &&
			    binned_spike_times[neuron][sample] <
			std::numeric_limits<Real>::max() -10) { max =
			binned_spike_times[neuron][sample];
			}*/
		}
		// std::cout << "Max - Min for Population " << pop.pid()<< ": "<<
		// max-min << ", " << max << ", "<<min<<std::endl;
		if (min == std::numeric_limits<Real>::max()) {
			// There was no spike
			for (size_t neuron = 0; neuron < binned_spike_times.size();
			     neuron++) {
				res[sample][neuron] = 0.0;
			}
		}
		else {
			for (size_t neuron = 0; neuron < binned_spike_times.size();
			     neuron++) {
				// Substract min --> first spike is zero, last spike might be
				// larger then duration
				Real tmp = binned_spike_times[neuron][sample] - min;
				if (tmp >= new_duration) {
					res[sample][neuron] = 0.0;
				}
				else {
					res[sample][neuron] = 1.0 - (tmp / new_duration);
					assert(res[sample][neuron] > 0);
				}
			}
		}
	}
	return res;
}

void conv_spikes_per_kernel(const std::string& filename, const PopulationBase& pop,
                            Real duration, Real pause, size_t batch_size, Real norm)
{
    auto res = spikes_to_rates(pop, duration, pause, batch_size, norm);
    std::ofstream file;
	file.open(filename);
	file << "Neuron1,Neuron2,Neuron3,Neuron4,max,sum\n";
	double neur1, neur2, neur3, neur4;
	for (size_t sample = 0; sample < res.size(); sample++) {
		for (size_t x = 0; x < 25; x+=2) {
			for (size_t y = 0; y < 25; y+=2) {
				for (size_t fil = 0; fil < 32; fil++) {
                    neur1 = res[sample][x*26*32 +y*32 +fil];
                    neur2 = res[sample][x*26*32 +(y+1)*32 +fil];
                    neur3 = res[sample][(x+1)*26*32 +y*32 +fil];
                    neur4 = res[sample][(x+1)*26*32 +(y+1)*32 +fil];
					if (neur1 != 0 || neur2 != 0 || neur3 != 0 || neur4 != 0) {
						file << neur1 << "," << neur2 << "," << neur3 << ","
						     << neur4 << ","
						     << std::max({neur1, neur2, neur3, neur4}) << ","
						     << neur1 + neur2 + neur3 + neur4 << "\n";
					}
				}
			}
		}
	}
}

std::vector<std::vector<Real>> spikes_to_rates(const PopulationBase pop,
                                               Real duration, Real pause,
                                               size_t batch_size, Real norm)
{
	std::vector<std::vector<Real>> res(batch_size,
	                                   std::vector<Real>(pop.size()));
	std::vector<std::vector<uint16_t>> binned_spike_counts;
	for (const auto &neuron : pop) {
		binned_spike_counts.push_back(
		    SpikingUtils::spike_time_binning<uint16_t>(
		        -pause * 0.5, batch_size * (duration + pause) - (pause * 0.5),
		        batch_size, neuron.signals().data(0)));
	}
	if (norm > 0.0) {
		for (size_t sample = 0; sample < batch_size; sample++) {
			for (size_t neuron = 0; neuron < binned_spike_counts.size();
			     neuron++) {
				res[sample][neuron] =
				    Real(binned_spike_counts[neuron][sample]) / norm;
			}
		}
	}
	else {
		for (size_t sample = 0; sample < batch_size; sample++) {
			for (size_t neuron = 0; neuron < binned_spike_counts.size();
			     neuron++) {
				res[sample][neuron] = Real(binned_spike_counts[neuron][sample]);
			}
		}
	}
	return res;
}

std::vector<std::vector<std::vector<cypress::Real>>> getSpikeTimes(std::vector<PopulationBase> &populations, Real duration,
																						  Real pause, size_t batch_size)
{
	std::vector<std::vector<cypress::Real>> oneLayer;
	for(size_t j=0; j<populations.size(); j++)
	{
		oneLayer.push_back(std::vector<cypress::Real>(populations[j].size(), 0));
	}
	std::vector<std::vector<std::vector<cypress::Real>>> spike_times(batch_size, oneLayer); //[sample][layer][neuron]

	for(size_t layer=0; layer<populations.size(); layer++)
	{
//		for(const auto &neuron : populations[layer]) {
		for(size_t i=0; i<populations[layer].size(); i++) {
			std::vector<Real> binnedSpikes=SpikingUtils::spike_time_binning_TTFS(
				0.0, Real(batch_size) * (duration + pause), batch_size,
				populations[layer][i].signals().data(0)); //returns the first spike time for each sample in this neuron

			for(size_t k=0; k<batch_size; k++)
			{
				if(binnedSpikes[k]==std::numeric_limits<Real>::max())
				{
					spike_times[k][layer][i]=1;
				}
				else {
					spike_times[k][layer][i]=(binnedSpikes[k]-cypress::Real(k)*(duration+pause))/(duration+pause);
				}

					#ifndef NDEBUG
				assert(spike_times[k][layer][i]>=0 && spike_times[k][layer][i]<=1);
					#endif
			}
			//each spike is now at a time between 0 and 1
		}
	}
	return spike_times;
																						  }

size_t compare_labels(std::vector<uint16_t> &label, std::vector<uint16_t> &res)
{
	size_t count_correct = 0;
	if (label.size() > res.size()) {
		throw std::runtime_error("label data has incorrect size! Target: " +
		                         std::to_string(label.size()) +
		                         " Result: " + std::to_string(res.size()));
	}
	for (size_t i = 0; i < label.size(); i++) {
		if (label[i] == res[i]) {
			count_correct++;
		}
	}
	return count_correct;
}

std::vector<Real> av_pooling_image(std::vector<Real> &image, size_t height,
                                   size_t width, size_t pooling_size)
{
	size_t new_h = std::floor(Real(height) / Real(pooling_size));
	size_t new_w = std::floor(Real(width) / Real(pooling_size));
	std::vector<Real> res(new_h * new_w, 0.0);

	for (size_t h = 0; h < new_h; h++) {
		for (size_t w = 0; w < new_w; w++) {
			std::vector<Real> vals(pooling_size * pooling_size, 0.0);
			for (size_t h2 = 0; h2 < pooling_size; h2++) {
				for (size_t w2 = 0; w2 < pooling_size; w2++) {
					if ((h * pooling_size + h2) < height &&
					    (w * pooling_size + w2) < width) {
						vals[h2 * pooling_size + w2] =
						    image[(h * pooling_size + h2) * width +
						          w * pooling_size + w2];
					}
					else {
						vals[h2 * pooling_size + w2] = 0.0;
					}
				}
			}
			// res[h * new_w + w] = *std::max_element(vals.begin(), vals.end());
			res[h * new_w + w] =
			    std::accumulate(vals.begin(), vals.end(), 0.0) /
			    Real(vals.size());
		}
	}
	return res;
}

MNIST_DATA scale_mnist(MNIST_DATA &data, size_t pooling_size)
{
	MNIST_DATA res;
	std::get<1>(res) = std::get<1>(data);
	auto &tar_images = std::get<0>(res);
	for (auto &image : std::get<0>(data)) {
		tar_images.emplace_back(av_pooling_image(image, 28, 28, pooling_size));
	}
	return res;
}

SPIKING_MNIST read_data_to_spike(const size_t num_images, bool train_data,
                                 const Real duration, const Real max_freq,
                                 bool poisson, bool scale_down)
{
	mnist_helper::MNIST_DATA data;
	if (train_data) {
		data = mnist_helper::loadMnistData(num_images, "train");
	}
	else {
		data = mnist_helper::loadMnistData(num_images, "t10k");
	}
	if (scale_down) {
		data = scale_mnist(data);
	}

	return mnist_helper::mnist_to_spike(data, duration, max_freq, poisson);
}

std::vector<LocalConnection> conns_from_mat(
    const cypress::Matrix<Real> &weights, Real delay, Real scale_factor)
{
	std::vector<LocalConnection> res;
	if (scale_factor > 0) {
		for (size_t i = 0; i < weights.rows(); i++) {
			for (size_t j = 0; j < weights.cols(); j++) {
				res.emplace_back(
				    LocalConnection(i, j, weights(i, j) * scale_factor, delay));
			}
		}
		return res;
	}
	for (size_t i = 0; i < weights.rows(); i++) {
		for (size_t j = 0; j < weights.cols(); j++) {
			res.emplace_back(LocalConnection(i, j, weights(i, j), delay));
		}
	}
	return res;
}

void update_conns_from_mat(const std::vector<cypress::Matrix<Real>> &weights,
                           Network &netw, Real delay, Real scale_factor)
{
	for (size_t i = 0; i < weights.size(); i++) {
		netw.update_connection(Connector::from_list(conns_from_mat(
		                           weights[i], delay, scale_factor)),
		                       ("dense_" + std::to_string(i)).c_str());
	}
}
}  // namespace mnist_helper
