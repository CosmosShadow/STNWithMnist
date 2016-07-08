-- model cnn_location

local function createModel()
    local nPreviousOutputPlane

    local Convolution = nn.SpatialConvolution
    local Avg = nn.SpatialAveragePooling
    local ReLU = nn.ReLU
    local Max = nn.SpatialMaxPooling
    local SBatchNorm = nn.SpatialBatchNormalization

    -- 层间直连
    local function shortcut(nInputPlane,  nOutputPlane,  stride)
        if nInputPlane ~= nOutputPlane or stride ~= 1 then
            return nn.Sequential()
                :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
                :add(SBatchNorm(nOutputPlane))
        else
            return nn.Identity()
        end
    end

    -- 残差模块
    local function residualBlock(nOutputPlane, stride)
        local nInputPlane = nPreviousOutputPlane
        nPreviousOutputPlane = nOutputPlane

        local s = nn.Sequential()
        s:add(Convolution(nInputPlane, nOutputPlane, 3, 3, stride, stride, 1, 1))
        s:add(SBatchNorm(nOutputPlane))
        s:add(ReLU(true))
        s:add(Convolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        s:add(SBatchNorm(nOutputPlane))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(shortcut(nInputPlane, nOutputPlane, stride)))
            :add(nn.CAddTable(true))
            :add(ReLU(true))
    end

    -- 堆叠残差模块
    local function stackResidualBlock(repeatCount, nOutputPlane, stride)
        local seq = nn.Sequential()
        for i=1, repeatCount do
            seq:add(residualBlock(nOutputPlane, i == 1 and stride or 1))
        end
        return seq
    end

    -- parameters
    stackDepth = 1
    nPreviousOutputPlane = 16

    -- 定位
    local loc_net = nn.Sequential()
    loc_net:add(Convolution(1, 16, 3, 3, 1, 1, 1, 1))
    loc_net:add(ReLU(true))
    loc_net:add(stackResidualBlock(stackDepth, 16, 1))    --64
    loc_net:add(stackResidualBlock(stackDepth, 32, 2))    --32
    loc_net:add(stackResidualBlock(stackDepth, 64, 2))    --16
    loc_net:add(stackResidualBlock(stackDepth, 128, 2))   --8
    loc_net:add(stackResidualBlock(stackDepth, 128, 2))   --4
    loc_net:add(stackResidualBlock(stackDepth, 128, 2))   --2
    loc_net:add(nn.View(128*4):setNumInputDims(3))
    loc_net:add(nn.Linear(128*4, 64))
    loc_net:add(ReLU(true))
    loc_net:add(nn.Linear(64, 6))

    -- 空间变换网络
    local ct = nn.ConcatTable()
    local branch1 = nn.Sequential()
    branch1:add(nn.Transpose({3,4},{2,4}))
    local branch2 = nn.Sequential()
    branch2:add(loc_net)
    branch2:add(nn.AffineTransformMatrixGenerator(false, false, false))
    branch2:add(nn.AffineGridGeneratorBHWD(32, 32))
    ct:add(branch1)
    ct:add(branch2)
    local st = nn.Sequential()
    st:add(ct)
    st:add(nn.BilinearSamplerBHWD())
    st:add(nn.Transpose({2,4},{3,4}))

    -- parameter
    stackDepth = 1
    nPreviousOutputPlane = 16

    -- model
    local classifier = nn.Sequential()
    classifier:add(Convolution(1, 16, 3, 3, 1, 1, 1, 1))
    classifier:add(ReLU(true))
    classifier:add(stackResidualBlock(stackDepth, 32, 2))    --32
    classifier:add(stackResidualBlock(stackDepth, 64, 2))    --16
    classifier:add(stackResidualBlock(stackDepth, 128, 2))   --8
    classifier:add(stackResidualBlock(stackDepth, 128, 2))   --4
    classifier:add(stackResidualBlock(stackDepth, 128, 2))   --2
    classifier:add(nn.View(128*4):setNumInputDims(3))
    classifier:add(nn.Linear(128*4, 64))
    classifier:add(ReLU(true))
    classifier:add(nn.Linear(64, 10))
    classifier:add(nn.LogSoftMax())

    local model = nn.Sequential()
    model:add(st)
    model:add(classifier)

    -- 初始化参数
    local function ConvInit(name)
        for k,v in pairs(model:findModules(name)) do
            local n = v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
            v.bias:zero()
        end
    end
    -- 全部来一遍
    for k, param in ipairs(model:parameters()) do
        param:uniform(-0.1, 0.1)
    end
    -- 卷积
    ConvInit('nn.SpatialConvolution')
    -- 线性层
    for k,v in pairs(model:findModules('nn.Linear')) do
        v.bias:zero()
    end

    return model
end

model = createModel()









