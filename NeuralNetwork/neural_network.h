#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <numeric>
#include <vector>
#include <algorithm>

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

	neuron_net_system(const link_coeffs& _coeffs) : _coeffs(_coeffs) {

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

	size_t steps_donw() const {
		return _steps_done;
	}
private:
	size_t _steps_done;
};

#endif /* NEURAL_NETWORK_H */
