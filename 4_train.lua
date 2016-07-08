-- train

----------------------------------------------------------------------
print '==> defining training procedure'

function train()
    model:training()
    parameters, gradParameters = model:getParameters()

    local total_error= 0
    local right_count = 0

    local trsize = trainData.data:size()[1]

    for t = 1, global_iters_each_epochs do
        local inputs, targets = load_input_target_train()
        if global_use_cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()

            -- forward, backward
            local outputs = model:forward(inputs)
            local error = criterion:forward(outputs, targets)
            local grad = criterion:backward(outputs, targets)
            model:backward(inputs, grad)

            -- normalize
            local batchSize = inputs:size()[1]
            gradParameters:div(batchSize)

            if bPrintPredict then
                local random_index = 1
                local source_labels = target2labels(targets, output_terminal)
                local predicted_labels = prediction2labels(outputs, rnn_output_interval, output_terminal)
                print('==>target: '..source_labels[random_index]..' predict: '..predicted_labels[random_index]..' error: '..error)
            end

            -- -- 看看图片
            -- local source_labels = target2labels(targets, output_terminal)
            -- local predicted_labels = prediction2labels(outputs, rnn_output_interval, output_terminal)
            -- inputs = inputs:float()
            -- for i=1,global_batch_size do
            --     local name = i..'_'..source_labels[i]..'_'..predicted_labels[i]..'.png'
            --     image.save('/home/server/work/share/test/'..name, inputs[i])
            -- end
            -- os.exit()

            total_error= total_error+ error
            right_count = right_count + caculateRightCount(outputs, targets, rnn_output_interval, output_terminal)

            return error, gradParameters
        end

        -- optimize on current mini-batch
        optimMethod(feval, parameters, optimState)
    end

    if optimMethod == optim.rprop then
        print('==> loss:', total_error/(global_iters_each_epochs*optimState.niter), ' right accuracy:', right_count/(global_batch_size*(global_iters_each_epochs*optimState.niter)))
    else
        print('==> loss:', total_error/global_iters_each_epochs, ' right accuracy:', right_count/(global_batch_size*global_iters_each_epochs))
    end

    return f;
end

