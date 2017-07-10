require 'torch'
require 'nn'

require 'VanillaRNN'
require 'LSTM'

local utils = require 'util.utils'
local stack = require 'util.stack'
local re = require 'util.luaregex'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(kwargs)
  self.theme = utils.get_kwarg(kwargs, 'model_type')
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
  if word_stack:getn() == 0 then
    return words
  end
  for i,v in pairs(word_stack._et) do 
    words = words..v.word

  end
  return words
end


function perplexity(word_stack)
  local perplexity = 0
  for i,v in pairs(word_stack._et) do 
    perplexity = perplexity + v.perplexity

  end
  return perplexity/word_stack:getn()
end

function sanitise(cmd_sample)

            cmd_sample = string.gsub(cmd_sample, "[%p@]", "")
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

function LM:get_stresses(input, stress_pattern, carmel, wfst) 
local cmd = ("echo '$foo' | "..carmel.."/carmel -sliOEQbk 10 "..wfst):gsub('$foo', input:upper())
local stress = os.capture(cmd)
if string.match(stress, "0 0 0 0 0 0 0 0 0 0") then return {} end
local current_patterns = {}
local pattern = ""
for i,v in pairs(split(stress, " ")) do 
             -- print(string.match(v, "%d+"))    
              if string.match(v, "%d+") then     
               
               local sub_pattern = string.sub(stress_pattern, (#stress_pattern-#pattern)+1)
              if sub_pattern == pattern then
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
  local perplexity = 0
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

       --if char is not supported, guess again
       if string.find(self:decode_string(next_char[1]), "[%=%/%\\%*()_%[%]%{%}?.\n;:%-!'\"\t]") then
          goto continue
       end
       if string.find(self:decode_string(next_char[1]), " ") then
          if #word == 0 then 
            goto continue
          end
       end
       perplexity = perplexity + 1/probs[next_char[1][1]]

    end

   -- sample_cache = sample_cache .. self:decode_string(next_char[1])
   -- sampled[{{}, {t, t}}]:copy(next_char)

   -- print(self:decode_string(next_char[1]))
    --print(sample)
        --word = word..self:decode_string(next_char[1])
       --  sampled[{{}, {t, t}}]:copy(next_char)
      --scores = self:forward(next_char)

     -- if delimiter
      
         sampled[{{}, {t, t}}]:copy(next_char)
      scores = self:forward(next_char)

      if string.find(self:decode_string(next_char[1]), "[ ?.;:,%-!']") then
        print("c-\""..self:decode_string(next_char[1]).."\"")
       -- local next_char = self:encode_string(" "):view(1, 1)

      --  word = word.." "
        -- sampled[{{}, {t, t}}]:copy(next_char)
        --scores = self:forward(next_char)
        word = word.." "
        --print("char : \""..self:decode_string(next_char[1]).."\"")
        break;
      else
      word = word..self:decode_string(next_char[1]) 
      end
      
     
      ::continue::
    end
   -- self:resetStates()
    print("Perplexity : "..perplexity/#word)
    return { word = word, perplexity = perplexity }

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
  local T = utils.get_kwarg(kwargs, 'length', 400)
  local start_text = utils.get_kwarg(kwargs, 'start_text', '')
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)
  local sample = utils.get_kwarg(kwargs, 'sample', 1)
  local temperature = utils.get_kwarg(kwargs, 'temperature', 1)
  local line_syllables = utils.get_kwarg(kwargs, 'line_syllables', 10)
  local carmel = utils.get_kwarg(kwargs, 'carmel', "")
  local wfst = utils.get_kwarg(kwargs, 'wfst', "")
  local backwards = utils.get_kwarg(kwargs, 'is_backward', 0)
  local word_stress_threshold = utils.get_kwarg(kwargs, 'min_word_stress', 0.5)
  local lines_output = utils.get_kwarg(kwargs, 'lines_output', 'results/test-'..word_stress_threshold..'.csv')..'-'..word_stress_threshold..'.csv'
  local prefix = string.upper(utils.get_kwarg(kwargs, 'prefix', ''))
  if backwards then
    prefix = string.reverse(prefix)
  end

  --Iambic stress pattern
  local stress_pattern = ''
  local foot = utils.get_kwarg(kwargs, 'stress_foot', '*/')
  while #stress_pattern<line_syllables do
   stress_pattern = stress_pattern..foot
  end 
  
  local sampled = torch.LongTensor(1, T)
  -- for storing working string 
  local sample_cache = ""

  self:resetStates()
  
  
  local _, next_char= nil, nil
  local scores

  local word_stack = Stack:Create()
  local lines = {}
  local current_sample = start_text
  --Retain string of previous lines to keep RNN primed
  local legacy_sample = ""
  local attempts, line_index = 0, 1
 
  for t = 0, T do
    
   --if verbose > 0 then
    print("Prefix: "..prefix)
    print("CS: "..current_sample)
   --end

    local truncated_legacy = string.sub(string.reverse(legacy_sample), -50)
    

    preamble = string.upper(truncated_legacy..prefix..current_sample)
    local word_sample = self:sample_new_word(preamble, 1000, verbose, temperature, scores, sample)

    -- if this is the first ever word
    if #current_sample == 0 then
      -- make sure it has more than 1 letter
      while #word_sample.word <= 2 do
    --    print("Word: "..word_sample.word)
       word_sample = self:sample_new_word(string.upper(truncated_legacy), 1000, verbose, temperature, scores, sample)
     -- attempts = attempts + 1
      end
      --print("1 : "..string.reverse(word_sample.word)) 
   end
    
    local new_word = prefix..word_sample.word
    local word = {word = new_word, perplexity = word_sample.perplexity}
  --  if verbose > 0 then
    
 -- end

    word_stack:push(word)

    local sanitised_line = sanitise(word_stack_to_string(word_stack))
    local line_perplexity = perplexity(word_stack)

    if backwards == 1 then
      sanitised_line = string.reverse(sanitised_line)
      print("Attempts : "..attempts.." - New word : \""..string.reverse(new_word).."\"")
    else
      print("Attempts : "..attempts.." - New word : \""..new_word.."\"")
    end

    local stresses = self:get_stresses(sanitised_line, stress_pattern, carmel, wfst)
   
   -- If there are no valid pathways through the line
    if count(stresses) == 0 then
      word_stack:pop(1)
      attempts = attempts + 1
      if attempts > 5 then 
       word_stack:pop(1) 
        attempts=0
      end
    else
      -- There are valid pathways through the line
      attempts = 0
      prefix = ""
      local most_likely_stress, stress_prob = argmax(stresses)

      local acceptable_stress_threshold = 1

      if word_stress_threshold > 0 then
        for x = 1, word_stack:getn() do
           acceptable_stress_threshold = acceptable_stress_threshold*word_stress_threshold
        end
      else
        acceptable_stress_threshold = 0
      end

      if stress_prob < acceptable_stress_threshold then
        -- Word is not 'iambic' (or other pattern) enough
        --if verbose > 0 then
        print("Rejecting \""..new_word.."\": Does not conform strongly enough to correct stress pattern")
        --end

       word_stack:pop(1)
        attempts = attempts + 1
        if attempts > 5 then 
          word_stack:pop(1) 
          attempts = 0
        end

      --if the most likely stress pattern is greater than our acceptable line length (with 1 syllable error), pop
      elseif #most_likely_stress > line_syllables+1 then
        print("Rejecting \""..new_word.."\": Line has too many syllables")
        word_stack:pop(1)
        attempts = attempts + 1
        if attempts > 5 then 
          word_stack:pop(1)
          attempts = 0
        end

      elseif math.abs(#most_likely_stress-line_syllables) <= 1 then

        if verbose > 0 then
        print("Line created - "..most_likely_stress.."="..stress_prob.."; Perplexity= "..line_perplexity.."\n "..sanitised_line)
        end

        local output = assert(io.open(lines_output, 'a'))
       -- io.output(output)
        sanitised_line = string.gsub(sanitised_line, "\"", "")
        output:write(sanitised_line..","..stress_prob..","..line_perplexity.."\n")
        legacy_sample = sanitised_line:upper().."\n"..legacy_sample:upper()
        io.close(output)
        lines[line_index] = sanitised_line
        line_index = line_index+1
        print(sanitised_line)

        prefix = word_stack._et[1].word
        if #prefix > 2 then
          prefix =string.sub(prefix, 0, 2)
        end
       -- print("Legacy: \""..legacy_sample.."\""
        word_stack:pop(word_stack:getn())
      end

    current_sample = word_stack_to_string(word_stack)
    

    end
    
    

    if verbose > 0 then
   print("\""..current_sample.."\"") 
   print(stresses)
    end

  end

  return self:decode_string(sampled[1])

end


function LM:clearState()
  self.net:clearState()
end
