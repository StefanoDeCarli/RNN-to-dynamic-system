function next_state = lstm_step(state, weights, functions)
    % Unpack state and weights
    u_t = state.u;  h_t = state.h;  c_t = state.c;
    W = weights.W;  R = weights.R;  b = weights.b;
    sigmoid = functions.sigmoid;    tanh_func = functions.tanh;

    % Compute the gates
    f_t = sigmoid(W.f * u_t + R.f * h_t + b.f);     % Forget gate
    i_t = sigmoid(W.i * u_t + R.i * h_t + b.i);     % Input gate
    g_t = tanh_func(W.g * u_t + R.g * h_t + b.g);   % Candidate memory
    o_t = sigmoid(W.o * u_t + R.o * h_t + b.o);     % Output gate

    % Compute next state
    c_t_next = f_t .* c_t + i_t .* g_t;         % Updated cell state
    h_t_next = o_t .* tanh_func(c_t_next);      % Updated hidden state

    % Pack outputs
    next_state = struct('c', c_t_next, 'h', h_t_next);
end