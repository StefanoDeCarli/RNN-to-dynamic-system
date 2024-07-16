function next_state = gru_step(state, weights, functions)
    % Unpack state and weights
    u_t = state.u;  h_t = state.h;
    W = weights.W;  R = weights.R;  b = weights.b;
    sigmoid = functions.sigmoid;    tanh_func = functions.tanh;

    % Compute the gates
    r_t = sigmoid(W.r * u_t + R.r * h_t + b.r);                     % Reset gate
    z_t = sigmoid(W.z * u_t + R.z * h_t + b.z);                     % Update gate
    h_candidate = tanh_func(W.h * u_t + r_t .* (R.h * h_t) + b.h);  % Candidate hidden state
    
    % Compute next state
    h_t_next = (1 - z_t) .* h_candidate + z_t .* h_t;   % Updated hidden state

    % Pack outputs
    next_state = struct('h', h_t_next);
end