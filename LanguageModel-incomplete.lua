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
    s = s .. token
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

  local stress_pattern = re.compile("^\\*?(\\/\\*)+\\/?$")

  local sampled = torch.LongTensor(1, T)
  -- for storing working string 
  local sample_cache = ""
  self:resetStates()

  local scores, first_t
  if #start_text > 0 then
    if verbose > 0 then
      print('Seeding with: "' .. start_text .. '"')
    end
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2)
    sampled[{{}, {1, T0}}]:copy(x)
    scores = self:forward(x)[{{}, {T0, T0}}]
    first_t = T0 + 1
  else
    if verbose > 0 then
      print('Seeding with uniform probabilities')
    end
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
    first_t = 1
  end
  
  local _, next_char= nil, nil
  local delimiter_stack = Stack:Create()
  local current_pattern_stack = Stack:Create()
  --delimiter_stack:push(0)
  current_pattern_stack:push(1)

  local current_patterns = {}
  local attempts = 0
  current_patterns[1] = "test"

  for t = first_t, T do
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

      
    sample_cache = sample_cache .. self:decode_string(next_char[1])
    sampled[{{}, {t, t}}]:copy(next_char)

      local last_delimiter = delimiter_stack:peek(1)

      if string.find(self:decode_string(next_char[1]), "[ ?.;:\n,%-!']") then
     
      local last_word = ""
     -- if last_delimiter then string.sub(sample_cache, last_delimiter, t) end 

            last_word = trim(string.gsub(last_word, "%p", ""))
      
      local cmd_sample = string.gsub(trim(sample_cache), "[%?%.!]", "")
            cmd_sample = string.gsub(cmd_sample, "[\n,;:'%-]", " ")
            cmd_sample = string.gsub(cmd_sample, " ", "\" \"")
            cmd_sample = "\""..cmd_sample:upper().."\""
            cmd_sample = string.gsub(cmd_sample, "\"\" ", "")
            cmd_sample = string.gsub(cmd_sample, " \"\"", "")
        
      local cmd = ("echo '$foo' | /Users/jack/Documents/workspace/Poebot/graehl/carmel/bin/macosx/carmel -sliOEQbk 10 /Users/jack/Documents/workspace/Poebot/graehl/carmel/bin/macosx/wfst005.full.txt"):gsub('$foo', cmd_sample:upper())
--print(cmd)
      
      local stress = os.capture(cmd)

          -- retreat if the there is no stress parse (incorrect spelling) or if there are no permissible stresses
          local found = true
          if string.match(stress, "0 0 0 0 0 0 0 0 0 0") then found = false end

          -- THIS IS NOT WORKING CORRECTLY - REFACTOR STACK PATTERN
          local formed = false

          print("Pattern stack : "..current_pattern_stack:getn())
          print("Delimiter stack : "..delimiter_stack:getn())

          if current_pattern_stack:getn() == 0 then
            formed = true
          elseif not current_pattern_stack:peek(1) == 0 then
            formed = true
          end

          local too_many_attempts = false
          if attempts > 50 then too_many_attempts = true end
          
          print("Retreat : "..tostring(retreat).." (formed-"..tostring(formed)..", found-"..tostring(found)..", attempts-"..tostring(attempts)..", too many?-"..tostring(too_many_attempts)..")")
         
          local retreat = false

          if too_many_attempts then
            attempts = 0
            
            retreat = true
          else
            attempts = attempts + 1
          end

          if not formed then 
            retreat = true 
          end
          if not found then
           retreat = true 
          end
        --  if delimiter_stack:getn() == 0 then 
            -- retreat = true
         --   delimiter_stack:push(0)
         -- end

          if retreat then
            last_delimiter = delimiter_stack:pop(1)
            current_pattern_stack:pop(1)
            if not last_delimiter then last_delimiter = 0 end
            print("Was -- \""..sample_cache.."\"")
            print("Ret -- \""..string.sub(sample_cache, last_delimiter+1, #sample_cache).."\"")--..last_delimiter.." - "..#current_patterns)
            
            print("Is  -- \""..string.sub(sample_cache, 0 , last_delimiter).."\"")
            sample_cache = string.sub(sample_cache, 0, last_delimiter)
            print(t)
          else

        -- construct an object representing the stress probabilities of the current line    
          local pattern = "";
          local stress_count = 0;
          current_patterns = {}
            for i,v in pairs(split(stress, " ")) do 
              stress_count = stress_count+1
             -- print(string.match(v, "%d+"))    
              if string.match(v, "%d+") then     

              if (stress_pattern:match(pattern)) then
                print("Pattern: "..pattern)
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

        --  if not retreat then
          print("#Possible stress patterns : "..stress_count.." - "..#current_patterns)
          print(current_patterns)
          attempts = 0

         
          current_pattern_stack:push(#current_patterns)
          delimiter_stack:push(#sample_cache);  
         -- print(delimiter_stack:list())
         -- end

           --checkpoint_index = t; 
           
 
          -- current_pattern_stack:push(current_patterns);

        end
       
      --sample_cache = string.sub(sample_cache, 0, last_delimiter)

   
      
    else
      print(self:decode_string(next_char[1]))
    end

    scores = self:forward(next_char)
  end

   
  self:resetStates()
  return self:decode_string(sampled[1])
end


function LM:clearState()
  self.net:clearState()
end
