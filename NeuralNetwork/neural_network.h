#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <numeric>
#include <vector>
#include <algorithm>
#include <list>
#include <fstream>

using namespace std;

class Neuron {
public:
	enum class state {
		LOWER_STATE = -1, UPPER_STATE = 1
	};

	typedef double coeff_t;
	typedef state state_t;

	static state read(char c) {
		return c == '*' ? state::UPPER_STATE : state::LOWER_STATE;
	}

	static char write(state s) {
		return s == state::LOWER_STATE ? ' ' : '*';
	}

	template<typename _Iv, typename _Ic>
	static state_t calculate(_Iv val_b, _Iv val_e, _Ic coeff_b) {
		auto val = inner_product(val_b, val_e, coeff_b, coeff_t(0));
		return val > 0 ? state_t::UPPER_STATE : state_t::LOWER_STATE;
	}
};

typedef Neuron neuron_t;
typedef neuron_t::state_t state_t;
typedef vector<state_t> neurons_line;
typedef vector<vector<neuron_t::coeff_t>> link_coeffs;

class neuron_net_system {
public:
	const link_coeffs& _coeffs;

	struct _neurons_line {
		const neurons_line& _line;
		const size_t _width;
		const size_t _height;

		_neurons_line(
			const neurons_line& _line,
			size_t _width,
			size_t _height
		) : _line(_line),
			_width(_width),
			_height(_height) {

		}
	};

	neuron_net_system(const link_coeffs& _coeffs) : _coeffs(_coeffs) {

	}

	friend ostream& operator<<(ostream& stream, const _neurons_line& line) {
		neurons_line::const_iterator it = line._line.begin();
		for (size_t i = 0; i < line._height; i++) {
			for (size_t j = 0; j < line._width; j++) {
				stream << neuron_t::write(*it);
				it++;
			}
			stream << endl;
		}

		return stream;
	}

	bool do_step(neurons_line& line) {
		bool value_changed = false;

		neurons_line old_values(line.begin(), line.end());
		link_coeffs::const_iterator it_coeffs = _coeffs.begin();

		transform(line.begin(), line.end(), line.begin(),
			[&old_values, &it_coeffs, &value_changed](state_t old_value) -> state_t {
				auto new_value = neuron_t::calculate(
					old_values.begin(),
					old_values.end(),
					begin(*it_coeffs++)
				);
				value_changed = (new_value != old_value) || value_changed;
				return new_value;
			}
		);

		return value_changed;
	}

	size_t _do(neurons_line& line) {
		bool need_continue = true;
		_steps_done = 0;
		
		while (need_continue) {
			need_continue = do_step(line);
			++_steps_done;
		}

		return _steps_done;
	}

	link_coeffs learn_neuron_net(const list<neurons_line>& src_image) {
		link_coeffs result_coeffs;
		size_t neurons_count = src_image.front().size();

		result_coeffs.resize(neurons_count);
		for (size_t i = 0; i < neurons_count; i++) {
			result_coeffs[i].resize(neurons_count, 0);
		}

		for (size_t i = 0; i < neurons_count; i++) {
			for (size_t j = 0; j < i; j++) {
				neuron_t::coeff_t val = 0;
				val = accumulate(
					src_image.begin(),
					src_image.end(),
					neuron_t::coeff_t(0.0),
					[i, j](neuron_t::coeff_t old_val, const neurons_line& image) -> neuron_t::coeff_t {
						return old_val + static_cast<int>(image[i]) * static_cast<int>(image[j]);
					}
				);

				result_coeffs[i][j] = val;
				result_coeffs[j][i] = val;
			}
		}

		return result_coeffs;
	}

	size_t steps_done() const {
		return _steps_done;
	}
private:
	size_t _steps_done;
};

#endif /* NEURAL_NETWORK_H */
