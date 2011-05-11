#! /bin/bash

rm -f obj/test_input_*.txt.*out obj/make_cuda.o* obj/verify_output.txt

qsub jobscripts/job_script.make_cuda

while [ "`qstat | wc -l 2>/dev/null`" -gt "0" ]; do 
	echo -n '.'
	sleep 1
done

( 
	bin/verify_outputs.sh 
	echo "done verifying outputs"
)&>obj/verify_output.txt &

vim obj/make_cuda.o*
