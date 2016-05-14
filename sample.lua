require 'torch'
require 'nn'

require 'LanguageModel'

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
cmd:option('-length', 2000)
cmd:option('-start_text', '')
cmd:option('-sample', 1)
cmd:option('-temperature', 1)
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-verbose', 0)
cmd:option('-stress_regex', '^\\*?(\\/\\*)+\\/?$')
cmd:option('-line_syllables', 10)

cmd:option('-is_backward', 0)
cmd:option('-min_word_stress', 0.5)
cmd:option('-lines_output', 'results/test_results2.csv')
cmd:option('-carmel', '/Users/jack/Documents/workspace/Poebot/graehl/carmel/bin/macosx')
cmd:option('-wfst', '/Users/jack/Documents/workspace/Poebot/torch-rnn/torch-rnn/wfst005.txt')
local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local msg
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  model:cuda()
  msg = string.format('Running with CUDA on GPU %d', opt.gpu)
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  model:cl()
  msg = string.format('Running with OpenCL on GPU %d', opt.gpu)
else
  msg = 'Running in CPU mode'
end
if opt.verbose == 1 then print(msg) end

model:evaluate()

local sample = model:sample(opt)
print(sample)
