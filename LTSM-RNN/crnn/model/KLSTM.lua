--This module creates the described computational graph
require 'nn'
require 'nngraph'

local LSTM = {}

function LSTM.lstm(input_size, rnn_size)
--Definition of some of these functions:
--nn.Narrow(dim, start, len) - selects a subvector along 
--dim dimension having len elements starting from start index
    
--nn.CMulTable() - outputs the product of tensors in forwarded table
--nn.CAddTable() - outputs the sum of tensors in forwarded table
  --------------------- input structure ---------------------
   
  ------------------------------------------------------------  
  ----Step 1, as shown in the above computational graphs
  ------------------------------------------------------------
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- network input
  table.insert(inputs, nn.Identity()())   -- c at time t-1
  table.insert(inputs, nn.Identity()())   -- h at time t-1
  local input = inputs[1]
  local prev_c = inputs[2]
  local prev_h = inputs[3]

  --------------------- preactivations ----------------------
  local i2h = nn.Linear(input_size, 4 * rnn_size)(input)   -- input to hidden
  local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)    -- hidden to hidden
  local preactivations = nn.CAddTable()({i2h, h2h})        -- i2h + h2h

  
    
  ----Step 2, as shown in the above computational graphs 
  ------------------ non-linear transforms ------------------
  -- gates

  local pre_sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(preactivations)
  local all_gates = nn.Sigmoid()(pre_sigmoid_chunk)

  -- input
  local in_chunk = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(preactivations)
  local in_transform = nn.Tanh()(in_chunk)

  ---------------------- gate narrows -----------------------
  local in_gate = nn.Narrow(2, 1, rnn_size)(all_gates)
  local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(all_gates)
  local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(all_gates)
  --------------------------------------------------------------
    
  ---- Step 3, as in comp graph above---------------------------
  --------------------- next cell state ---------------------
  local c_forget = nn.CMulTable()({forget_gate, prev_c})  -- previous cell state contribution
  local c_input = nn.CMulTable()({in_gate, in_transform}) -- input contribution
  local next_c = nn.CAddTable()({
    c_forget,
    c_input
  })

  -------------------- next hidden state --------------------
  local c_transform = nn.Tanh()(next_c)
  local next_h = nn.CMulTable()({out_gate, c_transform})

  --------------------- output structure --------------------
  outputs = {}
  table.insert(outputs, next_c)
  table.insert(outputs, next_h)

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule(inputs, outputs)
end

return LSTM
