#! /bin/bash

for i in {1..3}; do
	for implementation in {seq,gpu,cublas}; do
		output="obj/test_input_${i}.txt.${implementation}.out"
		golden="inputs/test_input_${i}_golden.txt"
		if [ -f $output ]; then
			cmd="bin/diff_coo_matrices.py $output $golden"
			echo "$cmd"
			echo -n "$((`head -1 $output | tr ' ' '*'`)) elements. "
			if $cmd 2>/dev/null; then echo ""; fi
		else
			echo "*** no $implementation output for input $i"
		fi
		echo ""
	done
done
