#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <list>
#include <cmath>
#include <numeric>

using namespace std;

enum state{
	LOWER_STATE = -1, UPPER_STATE = 1
};

class Neuron {
private:
	state _state;

public:
	Neuron(state _state=LOWER_STATE) : _state(_state) {

	}

	static state read(uchar i) {
		return i >= 250 ? UPPER_STATE : LOWER_STATE;
	}

	static uchar write(state s) {
		return s == UPPER_STATE ? 255 : 0;
	}
};

class NeuronNet {
private:
	vector<Neuron> _neurons;
	vector<vector<float>> _synapses;
	const size_t _neurons_count;
	const size_t _width;
	const size_t _height;

	int _steps;

public:
	NeuronNet(
		const list<vector<state>>& etallons
	) : _width(sqrt(etallons.back().size())),
		_height(sqrt(etallons.back().size())),
		_neurons_count(etallons.back().size()) {

		learn_neuron_net(etallons);
	}

	size_t recognize(vector<state>& test_image) {
		bool need_continue = true;
		_steps = 0;

		while (need_continue) {
			need_continue = _do(test_image);
			_steps++;
		}

		return _steps;
	}

private:
	void learn_neuron_net(const list<vector<state>>& etallons) {
		vector<vector<float>> result_synapses(_neurons_count, vector<float>(_neurons_count, 0.0));

		for (size_t i = 0; i < _neurons_count; i++) {
			for (size_t j = 0; j < i; j++) {
				float syn_val = 0.0;
				syn_val = accumulate(
					etallons.begin(),
					etallons.end(),
					float(0.0),
					[i, j] (float old_syn_val, const vector<state>& image) {
					return old_syn_val + image[i] * image[j];
					}
				);
				result_synapses[i][j] = result_synapses[j][i] = syn_val;
			}
		}

		_synapses = result_synapses;
	}

	bool _do(vector<state>& image) {
		bool syn_val_changed = false;

		vector<state> noisy_image(image.begin(), image.end());
		vector<vector<float>>::const_iterator cit = _synapses.begin();

		transform(
			image.begin(),
			image.end(),
			image.begin(),
			[this, &noisy_image, &cit, &syn_val_changed](state old_status) -> state {
			state new_status = activate(
				noisy_image.begin(),
				noisy_image.end(),
				begin(*cit++)
			);
			syn_val_changed = (new_status != old_status) || syn_val_changed;
			return new_status;
			}
		);

		return syn_val_changed;
	}

	state activate(
		vector<state>::const_iterator first1,
		vector<state>::const_iterator last1,
		vector<float>::const_iterator first2
	) {
		float val = inner_product(
			first1,
			last1,
			first2,
			float(0.0)
		);
		return val > 0 ? UPPER_STATE : LOWER_STATE;
	}
};

#endif /* NEURAL_NETWORK_H */
