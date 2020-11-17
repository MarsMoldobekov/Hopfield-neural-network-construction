#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <numeric>
#include <vector>

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
		auto val = inner_product(
			val_b,
			val_e,
			coeff_b,
			coeff_t(0)
		);
		return val > 0 ? state_t::UPPER_STATE : state_t::LOWER_STATE;
	}
};

typedef Neuron neuron_t;
typedef neuron_t::state_t state_t;
typedef vector<state_t> neurons_line;
typedef vector<vector<neuron_t::coeff_t>> link_coeffs;

#endif /* NEURAL_NETWORK_H */
