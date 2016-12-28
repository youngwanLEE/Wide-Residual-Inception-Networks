--  Wide-Residual-Inception Neworks(WR-Inception Networks)
--  by Youngwan Lee

--  ************************************************************************
--  This code incorporates material from:

--  "Wide Residual Networks", http://arxiv.org/abs/1605.07146
--  authored by Sergey Zagoruyko and Nikos Komodakis

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************

local nn = require 'nn'
local utils = paths.dofile'utils.lua'

assert(opt and opt.depth)
assert(opt and opt.num_classes)
assert(opt and opt.widen_factor)

local Convolution = nn.SpatialConvolution
local DilatedConvolution = nn.SpatialDilatedConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function Dropout()
   return nn.Dropout(opt and opt.dropout or 0,nil,true)
end

local function BNReLUConv(nInputPlane, nOutputPlane, kh, kw, sh, sw, ph, pw) -- BN-ReLU-Conv
  assert(not (nInputPlane == nil)) 
  assert(not (nOutputPlane == nil))
  assert(not (kh == nil)) 
  assert(not (kw == nil)) 
  local sh = sh or 1
  local sw = sw or 1
  local ph = ph or 0
  local pw = pw or 0
  local layer = nn.Sequential():add(SBatchNorm(nInputPlane))
                               :add(ReLU(true))
                               :add(Convolution(nInputPlane, nOutputPlane, kh, kw, sh, sw, ph, pw))
                               
  return layer
end 

local function BNReLUDilatedConv(nInputPlane, nOutputPlane, kh, kw, sh, sw, ph, pw, dw, dh) -- BN-ReLU - Dilated Conv
  assert(not (nInputPlane == nil)) 
  assert(not (nOutputPlane == nil))
  assert(not (kh == nil)) 
  assert(not (kw == nil)) 
  local sh = sh or 1
  local sw = sw or 1
  local ph = ph or 0
  local pw = pw or 0
  local dw = dw or 1
  local dh = dh or 1
  local layer = nn.Sequential():add(SBatchNorm(nInputPlane))
                               :add(ReLU(true))
                               :add(DilatedConvolution(nInputPlane, nOutputPlane, kh, kw, sh, sw, ph, pw, dh, dw))
                               
  return layer
end 

local function createModel(opt)
   local depth = opt.depth

   local blocks = {}
   
   local function Tower(layers)
    local tower = nn.Sequential()
    for i=1,#layers do
      tower:add(layers[i])
    end
    return tower
   end   

   local function FilterConcat(towers)
      local concat = nn.DepthConcat(2)
      for i=1,#towers do
        concat:add(towers[i])
      end
      return concat
   end

   local function IdentityMapping(towers)
      local concatAdd = nn.Sequential()
      local shortCut = nn.Identity()
      concatAdd:add(nn.ConcatTable():add(towers)
                                    :add(shortCut)
                                    )
      concatAdd:add(nn.CAddTable()) 
      return concatAdd
   end

   local function IndentityFilterConcat(towers)
      local shortCut = nn.Identity()
      local concat = nn.DepthConcat(2)
      for i=1,#towers do
        concat:add(towers[i])
      end
      concat:add(shortCut)
      return concat
   end
   -- Wide-Residual-Inception Networks
   local function InceptionResnet_v2(fs_start) -- fig 16
     

     local path1 = nn.Identity()

     local fs2a = {256};                                    fs2a[0] = fs_start 
     local fs2b = {256};                                    fs2b[0] = fs2a[1]
     local fs2c = {128,256};                                fs2c[0] = fs2a[1]
     local fs2  = {fs_start};  --local fs2  = {384}; 
           fs2[0]  = fs2a[#fs2a] + fs2b[#fs2b] + fs2c[#fs2c] 
     
     local stem  = nn.Sequential():add(BNReLUConv(fs2a[0], fs2a[1], 1, 1, 1, 1, 0, 0))
     local branch1 = Tower(
                      {
                         BNReLUConv(fs2b[0], fs2b[1], 3, 3, 1, 1, 1, 1)
                      }
                    )
     local branch2 = Tower(
                      {
                         BNReLUConv(fs2c[0], fs2c[1], 3, 3, 1, 1, 1, 1),
                         BNReLUConv(fs2c[1], fs2c[2], 3, 3, 1, 1, 1, 1)
                      }
                    )
      local merge_branch = IndentityFilterConcat( {branch1, branch2} )
      stem:add(merge_branch)
      stem:add(BNReLUConv(fs2[0], fs2[1], 1, 1, 1, 1, 0, 0))
                                   
     local net = nn.Sequential()                                 
     net:add(nn.ConcatTable():add(path1)
                             :add(stem)
                             )
     net:add(nn.CAddTable()) 
    
     local fs_final = fs2[#fs2] 
     --assert(fs_final == fs_start)
    
     return net
   end

   local function InceptionResnet_v3(fs_start) -- fig 16
     
     local path1          = nn.Identity()
     local stem           = nn.Sequential()
     local branch         = nn.Sequential()                                     
     local branch_concat  = nn.DepthConcat(2)
     local path2b_sub     = nn.DepthConcat(2) 

     
     local start_dim = fs_start
     local fs_root   = {128}                                    fs_root[0]  = fs_start
     local fs2a      = {128};                                   fs2a[0]     = fs_start 
     local fs2b      = {64};                                    fs2b[0]     = fs_root[1]
     local fs2c      = {64,128};                                fs2c[0]     = fs_root[1]  
     local fs2       = {fs_start};  --local fs2  = {384}; 
           fs2[0]    = fs2a[#fs2a] + fs2b[#fs2b] + fs2c[#fs2c] 
     

      local path2a = BNReLUConv(fs2a[0],fs2a[1], 1, 1, 1, 1, 0, 0)

      local path2b = nn.Sequential():add(BNReLUConv(fs_root[0], fs_root[1], 1, 1, 1, 1, 0, 0))
      local path2b_sub1 = BNReLUConv(fs2b[0], fs2b[1], 3, 3, 1, 1, 1, 1)
      local path2b_sub2 = Tower(
                            {
                              BNReLUConv(fs2c[0], fs2c[1], 3, 3, 1, 1, 1, 1),
                              BNReLUConv(fs2c[1], fs2c[2], 3, 3, 1, 1, 1, 1)
                            }
                          )
      
      path2b_sub:add(path2b_sub1)
      path2b_sub:add(path2b_sub2)
      path2b:add(path2b_sub)

      branch_concat:add(path2a)
      branch_concat:add(path2b)

     branch:add(branch_concat)
     branch:add(BNReLUConv(fs2[0], fs2[1], 1, 1, 1, 1, 0, 0))
     
     stem:add(nn.ConcatTable():add(path1)
                              :add(branch)
                             )
     stem:add(nn.CAddTable()) 
    
     local fs_final = fs2[#fs2] 
     --assert(fs_final == fs_start)
    
     return stem
   end

   local function InceptionResnet_v3_Dilated(fs_start) -- fig 16
     
     local path1          = nn.Identity()
     local stem           = nn.Sequential()
     local branch         = nn.Sequential()                                     
     local branch_concat  = nn.DepthConcat(2)
     local path2b_sub     = nn.DepthConcat(2) 

     
     local start_dim = fs_start
     local fs_root   = {128}                                  fs_root[0]  = fs_start
     local fs2a      = {128};                                 fs2a[0]     = fs_start 
     local fs2b      = {256};                                 fs2b[0]     = fs_root[1]
     local fs2c      = {128};                                 fs2c[0]     = fs_root[1]  
     local fs2       = {fs_start};  --local fs2  = {384}; 
           fs2[0]    = fs2a[#fs2a] + fs2b[#fs2b] + fs2c[#fs2c] 
     

      local path2a = BNReLUConv(fs2a[0],fs2a[1], 1, 1, 1, 1, 0, 0)

      local path2b = nn.Sequential():add(BNReLUConv(fs_root[0], fs_root[1], 1, 1, 1, 1, 0, 0))
      local path2b_sub1 = BNReLUConv(fs2b[0], fs2b[1], 3, 3, 1, 1, 1, 1)
      local path2b_sub2 = BNReLUDilatedConv(fs2c[0], fs2c[1],3,3,1,1,1,1,2,2) -- 5x5 conv effect
      
      path2b_sub:add(path2b_sub1)
      path2b_sub:add(path2b_sub2)
      path2b:add(path2b_sub)

      branch_concat:add(path2a)
      branch_concat:add(path2b)

     branch:add(branch_concat)
     branch:add(BNReLUConv(fs2[0], fs2[1], 1, 1, 1, 1, 0, 0))
     
     stem:add(nn.ConcatTable():add(path1)
                              :add(branch)
                             )
     stem:add(nn.CAddTable()) 
    
     local fs_final = fs2[#fs2] 
     --assert(fs_final == fs_start)
    
     return stem
   end

   local function InceptionResnet_v1(fs_start) -- fig 16
     

     local path1 = nn.Identity()

     local start_dim = fs_start
     local fs2a = {128};                              fs2a[0] = fs_start 
     local fs2b = {64,128};                           fs2b[0] = fs2a[1]
     local fs2c = {64,128,64};                        fs2c[0] = fs2a[1]
     local fs2  = {fs_start};  --local fs2  = {384}; 
           fs2[0]  = fs2a[#fs2a] + fs2b[#fs2b] + fs2c[#fs2c] 
     
     local path2a  = nn.Sequential():add(BNReLUConv(fs2a[0], fs2a[1], 1, 1, 1, 1, 0, 0))
     local path2b = nn.Sequential():add(BNReLUConv(fs2b[0], fs2b[1], 1, 1, 1, 1, 0, 0))
                                   :add(BNReLUConv(fs2b[1], fs2b[2], 3, 3, 1, 1, 1, 1))

     local path2c = nn.Sequential():add(BNReLUConv(fs2c[0], fs2c[1], 1, 1, 1, 1, 0, 0))
                                   :add(BNReLUConv(fs2c[1], fs2c[2], 3, 3, 1, 1, 1, 1))
                                   :add(BNReLUConv(fs2c[2], fs2c[3], 3, 3, 1, 1, 1, 1))
     
     local path2  = nn.Sequential():add(nn.ConcatTable():add(path2a)
                                                        :add(path2b)
                                                        :add(path2c)
                                                        )
                                   :add(nn.JoinTable(2, 4))
                                   :add(BNReLUConv(fs2[0], fs2[1], 1, 1, 1, 1, 0, 0))
                                   
     local net = nn.Sequential()                                 
     net:add(nn.ConcatTable():add(path1)
                             :add(path2)
                             )
     net:add(nn.CAddTable()) 
    
     local fs_final = fs2[#fs2] 
     --assert(fs_final == fs_start)
    
     return net
   end

   local function InceptionResnetA_multiplicity(fs_start) -- fig 16
     

     local path1 = nn.Identity()

     local start_dim = 64
     local fs2a = {fs_start};                         fs2a[0] = fs_start 
     local fs2b = {start_dim,fs_start};               fs2b[0] = fs2a[1]
     local fs2c = {start_dim,fs_start,fs_start};      fs2c[0] = fs2a[1]
     local fs2  = {fs_start};  --local fs2  = {384}; 
           fs2[0]  = fs2a[#fs2a] + fs2b[#fs2b] + fs2c[#fs2c] 
     
     local path2a  = nn.Sequential():add(BNReLUConv(fs2a[0], fs2a[1], 1, 1, 1, 1, 0, 0))
     local path2b = nn.Sequential():add(BNReLUConv(fs2b[0], fs2b[1], 1, 1, 1, 1, 0, 0))
                                   :add(BNReLUConv(fs2b[1], fs2b[2], 3, 3, 1, 1, 1, 1))

     -- local path2c = nn.Sequential():add(BNReLUConv(fs2c[0], fs2c[1], 1, 1, 1, 1, 1, 1))
     --                               :add(BNReLUConv(fs2c[1], fs2c[2], 3, 3, 1, 1, 1, 1))
     --                               :add(BNReLUConv(fs2c[2], fs2c[3], 3, 3, 1, 1, 1, 1))
     local path2c = IdentityMapping(
                  Tower(
                     {
                        BNReLUConv(fs2c[0], fs2c[1], 1, 1, 1, 1, 0, 0),
                        BNReLUConv(fs2c[1], fs2c[2], 3, 3, 1, 1, 1, 1),
                        BNReLUConv(fs2b[1], fs2b[2], 3, 3, 1, 1, 1, 1)
                     }
                  )
               )


     local path2  = nn.Sequential():add(nn.ConcatTable():add(path2a)
                                                        :add(path2b)
                                                        :add(path2c)
                                                        )
                                   :add(nn.JoinTable(2, 4))
                                   :add(BNReLUConv(fs2[0], fs2[1], 1, 1, 1, 1, 0, 0))
                                   
     local net = nn.Sequential()                                 
     net:add(nn.ConcatTable():add(path1)
                             :add(path2)
                             )
     net:add(nn.CAddTable()) 
    
     local fs_final = fs2[#fs2] 
     --assert(fs_final == fs_start)
    
     return net
   end

   
   local function wide_basic(nInputPlane, nOutputPlane, stride)
      local conv_params = {
         {3,3,stride,stride,1,1},
         {3,3,1,1,1,1},
      }
      local nBottleneckPlane = nOutputPlane

      local block = nn.Sequential()
      local convs = nn.Sequential()     

      convs:add(BNReLUConv(nInputPlane,nOutputPlane,3,3,stride,stride,1,1))
      convs:add(BNReLUConv(nOutputPlane,nOutputPlane,3,3,1,1,1,1))

      local shortcut = (stride == 1 and nInputPlane == nOutputPlane) and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)
     
      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   do
      assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth - 4) / 6

      local k = opt.widen_factor
      local nStages = torch.Tensor{64, 16*k, 32*k, 64*k}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(layer(wide_basic, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(wide_basic, nStages[2], nStages[4], n-1, 2)) -- Stage 2 (spatial size: 16x16)
      
      model:add(InceptionResnet_v2(nStages[4])) -- Inception Module
      
      model:add(layer(wide_basic, nStages[4], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4]))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], opt.num_classes))
   end

   utils.DisableBias(model)
   utils.testModel(model)
   utils.MSRinit(model)
   utils.FCinit(model)
   
   -- model:get(1).gradInput = nil

   return model
end

return createModel(opt)
