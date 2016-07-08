-- 

-- 更新加载入口
function load_method()
	load_global_parameters()
	load_method_sgd()
	-- load_method_rprop()
end

function load_global_parameters()
	-- GPU
	global_use_cuda = true
	global_GPU_device = 1	-- which one GPU	

	-- 总的训练数据
	global_train_count = 10000000

	-- 训练轮次控制
	global_iters_each_epochs = 50
	global_batch_size = 64

	-- 参数保存、加载
	global_save_parameter_iter = 20
	global_parameter_store_path = 'parameters'
	global_trained_parameter_path = ''
	-- global_trained_parameter_path = 'parameters/model_2016_07_01_15_38_11.t7'
end

function load_method_sgd()
	optimState = {
		learningRate = 0.1,
		learningRateDecay = 0,
		weightDecay = 0,
		momentum = 0.9,
		nesterov = true,
		dampening = 0,
	}
	optimMethod = optim.sgd
end

function load_method_rprop()
	optimState = {
		stepsize = 1e-5,
		etaplus = 1.2,
		etaminus = 0.5,
		stepsizemax = 1,
		stepsizemin = 1e-12,
		niter = 5
	}
	optimMethod = optim.rprop
end
