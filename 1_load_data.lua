-- load data

---------------------------------------------------------------------------------
print("==> Loading train data")

data_dir = 'mnist'
data_train_path = data_dir..'/train_32x32.t7'
data_test_path = data_dir..'/test_32x32.t7'

local data_trian = torch.load(data_train_path, 'ascii')
local data_test = torch.load(data_test_path, 'ascii')

function distort_image(img)
    local ns = 4
    -- 旋转
    local rot_angle = math.random()*math.pi/6 - math.pi/12
    img = image.rotate(img, rot_angle)
    -- 背景
    local dis_img = torch.ByteTensor(1, 64, 64):fill(0)
    -- copy
    dis_img:narrow(2, math.random(64-32), 32):narrow(3, math.random(64-32), 32):copy(img)
    -- 添加噪音
    for i=1,5 do
        local nosiy_region = dis_img:narrow(2, math.random(64-ns), ns):narrow(3, math.random(64-ns), ns)
        local nosiy_src = img:narrow(2, math.random(32-ns), ns):narrow(3, math.random(32-ns), ns)
        nosiy_region:copy(nosiy_src)
    end
    return dis_img
end

function load_input_target_train()

    local inputs = torch.zeros(global_batch_size, 1, 64, 64)
    local labels = torch.zeros(global_batch_size)

    for i=1,global_batch_size do
        local tsize = data_trian.data:size(1)
        local random_index = math.random(math.min(tsize, global_train_count))

        local img = data_trian.data[random_index]
        local dis_img = distort_image(img)
        inputs[i]:copy(dis_img)

        labels[i] = data_trian.labels[random_index]
    end

    -- 归一化处理
    inputs = inputs:div(255.0):add(-0.5)

    return inputs, labels
end







