--

require 'torch'
require 'nn'
require 'rnn'
require 'image'

-- train data
local temp = torch.load('mnist/train_32x32.t7', 'ascii')
print(temp)
img = temp.data[1]

is = 64
ns = 6

-- 旋转
img = image.rotate(img, math.pi/6)
bak_img = torch.ByteTensor(1, is, is):fill(0)

-- 扩大
bak_img:narrow(2, math.random(is-32), 32):narrow(3, math.random(is-32), 32):copy(img)

-- 添加噪音
for i=1,3 do
	nosiy_region = bak_img:narrow(2, math.random(is-ns), ns):narrow(3, math.random(is-ns), ns)
	nosiy_src = img:narrow(2, math.random(32-ns), ns):narrow(3, math.random(32-ns), ns)
	nosiy_region:copy(nosiy_src)
end


image.save('1.png', bak_img)
print(temp.labels:narrow(1, 1, 10))

