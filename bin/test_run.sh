#! /bin/bash

rm -f obj/test_input_*.txt.*out obj/make_{cuda,all}.o* obj/verify_output.txt obj/verify_all.o*

#qsub jobscripts/job_script.make_cuda
qsub jobscripts/job_script.make_all

while [ "`qstat | wc -l 2>/dev/null`" -gt "0" ]; do 
	echo -n '.'
	sleep 1
done

#( 
#    bin/verify_outputs.sh 
#    echo "done verifying outputs"
#)&>obj/verify_output.txt &
qsub jobscripts/job_script.verify_all

while ! tail -f obj/verify_all.o* 2>/dev/null; do sleep 0.1; done | grep -v ERROR
