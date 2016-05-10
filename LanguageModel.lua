require 'torch'
require 'nn'

require 'VanillaRNN'
require 'LSTM'

local utils = require 'util.utils'
local stack = require 'util.stack'
local re = require 'util.luaregex'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
 

  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  self.model_type = utils.get_kwarg(kwargs, 'model_type')
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
  self.num_layers = utils.get_kwarg(kwargs, 'num_layers')
  self.dropout = utils.get_kwarg(kwargs, 'dropout')
  self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size

  self.net = nn.Sequential()
  self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}

  self.net:add(nn.LookupTable(V, D))
  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    if self.model_type == 'rnn' then
      rnn = nn.VanillaRNN(prev_dim, H)
    elseif self.model_type == 'lstm' then
      rnn = nn.LSTM(prev_dim, H)
    end
    rnn.remember_states = true
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
    if self.batchnorm == 1 then
      local view_in = nn.View(1, 1, -1):setNumInputDims(3)
      table.insert(self.bn_view_in, view_in)
      self.net:add(view_in)
      self.net:add(nn.BatchNormalization(H))
      local view_out = nn.View(1, -1):setNumInputDims(2)
      table.insert(self.bn_view_out, view_out)
      self.net:add(view_out)
    end
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
  end

  -- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(H, V))
  self.net:add(self.view2)
end


function LM:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)

  for _, view_in in ipairs(self.bn_view_in) do
    view_in:resetSize(N * T, -1)
  end
  for _, view_out in ipairs(self.bn_view_out) do
    view_out:resetSize(N, T, -1)
  end

  return self.net:forward(input)
end


function LM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end


function LM:parameters()
  return self.net:parameters()
end


function LM:training()
  self.net:training()
  parent.training(self)
end


function LM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end


function LM:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end

-- Returns an enbedding of an input string
function LM:encode_string(s)
  local encoded = torch.LongTensor(#s)
  for i = 1, #s do
    local token = s:sub(i, i)
    local idx = self.token_to_idx[token]
    assert(idx ~= nil, 'Got invalid idx')
    encoded[i] = idx
  end
  return encoded
end

-- Takes a tensor and returns the string
function LM:decode_string(encoded)
  assert(torch.isTensor(encoded) and encoded:dim() == 1)
  local s = ''
  for i = 1, encoded:size(1) do
    local idx = encoded[i]
    local token = self.idx_to_token[idx]
    if token then
    s = s .. token
  end
  end
  return s
end

function os.capture(cmd, raw)
  local f = assert(io.popen(cmd, 'r'))
  local s = assert(f:read('*a'))
  f:close()
  if raw then return s end
  s = string.gsub(s, '^%s+', '')
  s = string.gsub(s, '%s+$', '')
  s = string.gsub(s, '[\n\r]+', ' ')
  return s
end

function trim(s)
  return s:gsub("^%s+", ""):gsub("%s+$", "")
end

function split(str, delim)
    local result,pat,lastPos = {},"(.-)" .. delim .. "()",1
    for part, pos in string.gfind(str, pat) do
        table.insert(result, part); lastPos = pos
    end
    table.insert(result, string.sub(str, lastPos))
    return result
end

function word_stack_to_string(word_stack)
  local words = ""
  for i,v in pairs(word_stack._et) do 
    words = words..v

  end
  return words
end

function sanitise(cmd_sample)

            cmd_sample = string.gsub(cmd_sample, "%p", "")
            cmd_sample = string.gsub(cmd_sample, "[\n,]", " ")
            cmd_sample = string.gsub(cmd_sample, " ", "\" \"")
            cmd_sample = "\""..cmd_sample:upper().."\""
            cmd_sample = string.gsub(cmd_sample, "\"\" ", "")
            cmd_sample = string.gsub(cmd_sample, " \"\"", "")

  return cmd_sample

end

function argmax(stresses)
  local best_prob = 0
  local best_stress = ""
  for i,v in pairs(stresses) do 
    if tonumber(v) > best_prob then
      best_prob = tonumber(v)
      best_stress = i
    end

  end
  return best_stress, best_prob
end

function count(object) 
  local count = 0
  for i,v in pairs(object) do 
    count = count + 1
  end
  return count
end

function LM:get_stresses(input, stress_pattern) 
local cmd = ("echo '$foo' | /Users/jack/Documents/workspace/Poebot/graehl/carmel/bin/macosx/carmel -sliOEQbk 10 /Users/jack/Documents/workspace/Poebot/graehl/carmel/bin/macosx/wfst005.full.txt"):gsub('$foo', input:upper())
local stress = os.capture(cmd)
if string.match(stress, "0 0 0 0 0 0 0 0 0 0") then return {} end
local current_patterns = {}
local pattern = ""
for i,v in pairs(split(stress, " ")) do 
             -- print(string.match(v, "%d+"))    
              if string.match(v, "%d+") then     
               
              if (stress_pattern:match(pattern)) then
               
                current_patterns[pattern] = v
              end

                if #pattern == 1 then 
                  current_patterns[pattern] = v
                end

                pattern = ""

              else
                pattern = pattern .. string.gsub(string.gsub(v, "S%*", "/"), "S", "*")              
              end
    end
return current_patterns;
end

function LM:sample_new_word(start_text, num, verbose, temperature, scores, sample)

local first_t
local sampled = torch.LongTensor(1, num)
  if #start_text > 0 then
    if verbose > 0 then
      print('Seeding with: "' .. start_text .. '"')
    end
    self:resetStates()
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2)
    --sampled[{{}, {1, T0}}]:copy(x)
    scores = self:forward(x)[{{}, {T0, T0}}]
    first_t = T0 + 1
    print("Start index : "..first_t)
  else
    if verbose > 0 then
      print('Seeding with uniform probabilities')
    end
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
    first_t = 1
  end


  local word = ""
  local _, next_char = nil, nil
   for t = first_t, num do
    -- If we haven't started sampling yet...
    if sample == 0 then
      _, next_char = scores:max(3)
      next_char = next_char[{{}, {}, 1}]
    else
      --Otherwise find a tensor of probabilities, exponentiated
       local probs = torch.div(scores, temperature):double():exp():squeeze()
       -- Normalise...
       probs:div(torch.sum(probs))
       -- Select 1 from a multinomial distribution
       next_char = torch.multinomial(probs, 1):view(1, 1)
    end

   -- sample_cache = sample_cache .. self:decode_string(next_char[1])
   -- sampled[{{}, {t, t}}]:copy(next_char)

   -- print(self:decode_string(next_char[1]))
    --print(sample)
        word = word..self:decode_string(next_char[1])
         sampled[{{}, {t, t}}]:copy(next_char)
      scores = self:forward(next_char)

     -- if delimiter
      if string.find(self:decode_string(next_char[1]), "[ ?.;:\n,%-!']") then
        break;
      end

     

    end
    self:resetStates()

    return word
end
--[[
Sample from the language model. Note that this will reset the states of the
underlying RNNs.

Inputs:
- init: String of length T0
- max_length: Number of characters to sample

Returns:
- sampled: (1, max_length) array of integers, where the first part is init.
--]]
function LM:sample(kwargs)
  local T = utils.get_kwarg(kwargs, 'length', 100)
  local start_text = utils.get_kwarg(kwargs, 'start_text', '')
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)
  local sample = utils.get_kwarg(kwargs, 'sample', 1)
  local temperature = utils.get_kwarg(kwargs, 'temperature', 1)
  local line_syllables = utils.get_kwarg(kwargs, 'line_syllables', 10)
  local lines_output = utils.get_kwarg(kwargs, 'lines_output', 'results/test.csv')


  --Iambic stress regex
  local stress_pattern = re.compile(utils.get_kwarg(kwargs, 'stress_regex', '^\\*?(\\/\\*)+\\/?$'))

  local sampled = torch.LongTensor(1, T)
  -- for storing working string 
  local sample_cache = ""
  self:resetStates()
  
  
  local _, next_char= nil, nil
  local scores

  local word_stack = Stack:Create()
  local current_sample = start_text

  local attempts = 0
 
  for t = 0, 10000 do
    
   

    local new_word = self:sample_new_word(current_sample, 1000, verbose, temperature, scores, sample)
    print("Attempts : "..attempts.." - New word : \""..new_word.."\"")
    word_stack:push(new_word)

    local sanitised_line = sanitise(word_stack_to_string(word_stack))
   
    local stresses = self:get_stresses(sanitised_line, stress_pattern)
   
    if count(stresses) == 0 then
      word_stack:pop(1)
      attempts = attempts + 1
      if attempts > 5 then word_stack:pop(1) end
    else
      attempts = 0

      local most_likely_stress, stress_prob = argmax(stresses)

      if #most_likely_stress > line_syllables then
        word_stack:pop(1)
        attempts = attempts + 1
        if attempts > 5 then word_stack:pop(1) end

      elseif #most_likely_stress == line_syllables then

        print("WE'VE MADE A LINE - "..most_likely_stress.."="..stress_prob..";\n "..sanitised_line)
        
        local output = assert(io.open(lines_output, 'a'))
       -- io.output(output)
        output:write(string.gsub(sanitised_line, "\"", "")..","..stress_prob.."\n")
        io.close(output)
        --io.output()
        current_sample = ""
        self:resetStates()
        word_stack:pop(word_stack:getn())
      end

    end
    current_sample = word_stack_to_string(word_stack)

   print("\""..current_sample.."\"") 
   print(stresses)
  end
   --print(sanitise(word_stack_to_string(word_stack)))

  return self:decode_string(sampled[1])

end


function LM:clearState()
  self.net:clearState()
end
