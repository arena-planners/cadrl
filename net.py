"""Self-contained torch reimplementation of GA3C-CADRL's value/policy network.

The upstream planner (``mit-acl/cadrl_ros``, the ICRA'17 / IROS'18 decentralized
collision-avoidance work, GA3C-CADRL variant) ships in-repo TensorFlow-1
checkpoints (``checkpoints/network_*.{meta,index,data-*}``). The deployed graph in
``scripts/network.py`` (``NetworkVP_rnn``, ``MULTI_AGENT_ARCH == 'RNN'``) is:

    x (1 + 4 + 10*7 = 75) = [ num_other_agents,
                              host(4)  = [dist_to_goal, heading_to_goal, pref_speed, radius],
                              other_i(7) x 10 = [px, py, vx, vy, radius, combined_radius, dist] ]
      x_normalized = (x - avg) / std                       # fixed constant vectors
      host_vec      = x_normalized[1:5]
      other_seq     = x_normalized[5:].reshape(10, 7)
      h = LSTM(other_seq, sequence_length=num_other_agents) # tf.nn.dynamic_rnn, 64 hidden
      layer1 = relu(dense(concat(host_vec(4), h(64)) -> 256))
      layer2 = relu(dense(layer1 -> 256))
      fc1    = relu(dense(layer2 -> 256))                   # 'fullyconnected1'
      logits_p = dense(fc1 -> num_actions=11);  policy = softmax(logits_p)
      logits_v = dense(fc1 -> 1)                            # state value

TF1 is *not* loaded at run time. The trained variables were dumped from the
checkpoint into ``model/cadrl_ga3c.npz`` (see ``weights.yaml`` / extraction
provenance) and this module reimplements the forward pass in torch.

Layout conversions performed at load time:
  * ``tf.layers.dense`` stores ``kernel`` as ``(in, out)`` -> transposed to torch
    ``(out, in)``.
  * The TF ``LSTMCell`` kernel is ``(in_dim + hidden, 4*hidden)`` with gate order
    ``(i, j, f, o)`` (input, new-input/cell, forget, output) and a hard-coded
    ``forget_bias = 1.0`` added to the forget pre-activation. torch's ``LSTMCell``
    uses gate order ``(i, f, g, o)`` and no extra forget bias, so we reorder the
    gate blocks, split the kernel into input/hidden halves, and fold ``+1.0`` into
    the forget bias. Verified bit-for-bit against the original TF graph.

The host-frame input state is built in ``planner.py`` (mirroring
``agent.Agent.observe``); this module only owns normalization + the network.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Config constants copied verbatim from upstream scripts/network.py:Config ---
HOST_AGENT_OBSERVATION_LENGTH = 4   # dist_to_goal, heading_to_goal, pref_speed, radius
OTHER_AGENT_OBSERVATION_LENGTH = 7  # px, py, vx, vy, radius, combined_radius, dist
MAX_NUM_OTHER_AGENTS_OBSERVED = 10
RNN_HIDDEN = 64
FIRST_STATE_INDEX = 1               # index 0 holds num_other_agents (RNN seq length)

FULL_STATE_LENGTH = (
    1
    + HOST_AGENT_OBSERVATION_LENGTH
    + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_OBSERVATION_LENGTH
)  # == 75

# Normalization vectors (Config.NN_INPUT_{AVG,STD}_VECTOR), reconstructed exactly.
_HOST_AVG = np.array([0.0, 0.0, 1.0, 0.5])
_HOST_STD = np.array([5.0, 3.14, 1.0, 1.0])
_OTHER_AVG = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0])
_OTHER_STD = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0])
_RNN_HELPER_AVG = np.array([0.0])
_RNN_HELPER_STD = np.array([1.0])

NN_INPUT_AVG_VECTOR = np.hstack(
    [_RNN_HELPER_AVG, _HOST_AVG, np.tile(_OTHER_AVG, MAX_NUM_OTHER_AGENTS_OBSERVED)]
).astype(np.float32)
NN_INPUT_STD_VECTOR = np.hstack(
    [_RNN_HELPER_STD, _HOST_STD, np.tile(_OTHER_STD, MAX_NUM_OTHER_AGENTS_OBSERVED)]
).astype(np.float32)


class CADRLNet(nn.Module):
    """GA3C-CADRL RNN value/policy net; weights loaded from the extracted npz."""

    def __init__(self, num_actions: int = 11) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.lstm = nn.LSTMCell(OTHER_AGENT_OBSERVATION_LENGTH, RNN_HIDDEN)
        self.layer1 = nn.Linear(HOST_AGENT_OBSERVATION_LENGTH + RNN_HIDDEN, 256)
        self.layer2 = nn.Linear(256, 256)
        self.fc1 = nn.Linear(256, 256)
        self.logits_p = nn.Linear(256, num_actions)
        self.logits_v = nn.Linear(256, 1)

        self.register_buffer("avg_vec", torch.from_numpy(NN_INPUT_AVG_VECTOR))
        self.register_buffer("std_vec", torch.from_numpy(NN_INPUT_STD_VECTOR))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.avg_vec) / self.std_vec

    def _run_lstm(self, other_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Replicate tf.nn.dynamic_rnn over (B, 10, 7) with per-row sequence_length.

        Mirrors the upstream masking: cells beyond ``sequence_length`` are skipped
        and the returned hidden state is the state *as of* the last valid step
        (state is frozen once a row finishes), matching dynamic_rnn semantics.
        """
        b = other_seq.shape[0]
        device = other_seq.device
        h = torch.zeros(b, RNN_HIDDEN, device=device, dtype=other_seq.dtype)
        c = torch.zeros(b, RNN_HIDDEN, device=device, dtype=other_seq.dtype)
        for t in range(MAX_NUM_OTHER_AGENTS_OBSERVED):
            h_new, c_new = self.lstm(other_seq[:, t, :], (h, c))
            # only update rows whose sequence is still running at step t
            active = (t < seq_len).to(other_seq.dtype).unsqueeze(1)
            h = active * h_new + (1.0 - active) * h
            c = active * c_new + (1.0 - active) * c
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 75) raw state -> (policy (B, num_actions), value (B,))."""
        num_other = x[:, 0]
        xn = self._normalize(x)
        host_vec = xn[:, FIRST_STATE_INDEX : FIRST_STATE_INDEX + HOST_AGENT_OBSERVATION_LENGTH]
        other_vec = xn[:, FIRST_STATE_INDEX + HOST_AGENT_OBSERVATION_LENGTH :]
        other_seq = other_vec.reshape(
            -1, MAX_NUM_OTHER_AGENTS_OBSERVED, OTHER_AGENT_OBSERVATION_LENGTH
        )
        h = self._run_lstm(other_seq, num_other)
        layer1 = F.relu(self.layer1(torch.cat([host_vec, h], dim=1)))
        layer2 = F.relu(self.layer2(layer1))
        fc1 = F.relu(self.fc1(layer2))
        logits_p = self.logits_p(fc1)
        policy = F.softmax(logits_p, dim=1)
        value = self.logits_v(fc1).squeeze(1)
        return policy, value

    @torch.no_grad()
    def predict_p(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]

    @torch.no_grad()
    def load_npz(self, path: str) -> "CADRLNet":
        w = np.load(path)

        def dense(layer: nn.Linear, name: str) -> None:
            # TF (in, out) -> torch (out, in)
            layer.weight.copy_(torch.from_numpy(np.ascontiguousarray(w[f"{name}.kernel"].T)))
            layer.bias.copy_(torch.from_numpy(np.ascontiguousarray(w[f"{name}.bias"])))

        dense(self.layer1, "layer1")
        dense(self.layer2, "layer2")
        dense(self.fc1, "fc1")
        dense(self.logits_p, "logits_p")
        dense(self.logits_v, "logits_v")

        # LSTM: TF kernel (in_dim+hidden, 4*hidden), gate order (i, j, f, o),
        # forget_bias=1.0 added internally. torch wants (i, f, g, o) with weight_ih
        # (4*hidden, in_dim) and weight_hh (4*hidden, hidden).
        k = w["lstm.kernel"]            # (71, 256)
        b = w["lstm.bias"].copy()       # (256,)
        in_dim = OTHER_AGENT_OBSERVATION_LENGTH
        hh = RNN_HIDDEN
        w_x = k[:in_dim, :]             # (7, 256) input->gates
        w_h = k[in_dim:, :]             # (64, 256) hidden->gates

        def reorder_cols(mat: np.ndarray) -> np.ndarray:
            # split into the 4 TF gate blocks (i, j, f, o) and reassemble as torch (i, f, g, o)
            i, j, f, o = (mat[:, g * hh : (g + 1) * hh] for g in range(4))
            return np.concatenate([i, f, j, o], axis=1)

        w_x = reorder_cols(w_x)         # (7, 256) -> torch col order
        w_h = reorder_cols(w_h)         # (64, 256)
        b = reorder_cols(b[None, :])[0] # (256,)
        # fold TF forget_bias=1.0 into the forget-gate bias block (2nd block now)
        b[hh : 2 * hh] = b[hh : 2 * hh] + 1.0

        # torch weight_ih/hh are (4*hidden, in/hidden) == transpose of our (in/hidden, 4*hidden)
        self.lstm.weight_ih.copy_(torch.from_numpy(np.ascontiguousarray(w_x.T)))
        self.lstm.weight_hh.copy_(torch.from_numpy(np.ascontiguousarray(w_h.T)))
        # torch has two bias vectors (bias_ih + bias_hh); put the full bias in ih, zero hh
        self.lstm.bias_ih.copy_(torch.from_numpy(np.ascontiguousarray(b)))
        self.lstm.bias_hh.zero_()

        self.eval()
        return self
