-- model cnn_location

local function createModel()
    local nPreviousOutputPlane

    local SBatchNorm = nn.SpatialBatchNormalization

    -- 层间直连
    local function shortcut(nInputPlane,  nOutputPlane,  stride)
        if nInputPlane ~= nOutputPlane or stride ~= 1 then
            return nn.Sequential()
                :add(nn.SpatialConvolution(nInputPlane, nOutputPlane, stride, stride, stride, stride))
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
        if nInputPlane ~= nOutputPlane or stride ~= 1 then
            s:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, stride, stride, stride, stride))
        else
            s:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        end
        s:add(SBatchNorm(nOutputPlane))
        s:add(nn.LeakyReLU(0.1, true))
        s:add(nn.SpatialConvolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        s:add(SBatchNorm(nOutputPlane))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(shortcut(nInputPlane, nOutputPlane, stride)))
            :add(nn.CAddTable())
            :add(nn.LeakyReLU(0.1, true))
    end

    -- 堆叠残差模块
    local function stackResidualBlock(repeatCount, nOutputPlane, stride)
        local seq = nn.Sequential()
        for i=1, repeatCount do
            seq:add(residualBlock(nOutputPlane, i == 1 and stride or 1))
        end
        return seq
    end

    --基础层:卷积 + 线性层
    -- 卷积
    local model_base = nn.Sequential()
    model_base:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))	--64*160
    model_base:add(nn.LeakyReLU(0.1, true))
    local stackDepth = 2
    nPreviousOutputPlane = 16
    model_base:add(stackResidualBlock(stackDepth, 32, 2))		--32*80
    model_base:add(stackResidualBlock(stackDepth, 64, 2))		--16*40
    model_base:add(stackResidualBlock(stackDepth, 128, 2))		--8*20
    model_base:add(stackResidualBlock(stackDepth, 256, 2))		--4*10
    model_base:add(stackResidualBlock(stackDepth, 512, 2))		--2*5
    -- 线性层
    model_base:add(nn.Reshape(512*2*5))
    model_base:add(nn.Linear(512*2*5, 1024))
    model_base:add(nn.LeakyReLU(0.1, true))
    -- 分类层: LSTM + 分类
    model_classifier = nn.Sequential()
    model_classifier:add(nn.LSTM(1024, 512))
    model_classifier:add(nn.LSTM(512, 512))
    model_classifier:add(nn.LSTM(512, 256))
    model_base:add(nn.LeakyReLU(0.1, true))
    model_classifier:add(nn.Linear(256, 64))
    model_base:add(nn.LeakyReLU(0.1, true))
    model_classifier:add(nn.Linear(64, noutputs))
    model_classifier:add(nn.LogSoftMax())
    -- 分类层重复
    local model_classifier_repeater = nn.Repeater(model_classifier, target_length)

    local model = nn.Sequential()
    model:add(model_base)
    model:add(model_classifier_repeater)

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







