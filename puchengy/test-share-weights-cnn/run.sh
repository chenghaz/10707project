for loss in 3 2 1; do
	for m in com vgg res; do
		for k in 16 32 64 128; do
			python SSDH-COM-VGG-RES.py --model $m --Loss $loss --K $k --save_path ./checkpoint/ssdh\_$loss\_$m\_$k
			python SSDH-COM-VGG-RES-binary.py --model $m --cp_path ./checkpoint/ssdh\_$loss\_$m\_$k --save_path ./checkpoint/binary\_$loss\_$m\_$k
			python SSDH-binary-acc.py --binary_path ./checkpoint/binary\_$loss\_$m\_$k
		done
	done
done