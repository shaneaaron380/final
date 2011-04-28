
use classes;
use calculations;
use body_list;

//command-line example 
//./a.out --iterations=10 --timestep=20 --input=infile --output=outfile
config const iterations = 1;
config const timestep = 1;
config const input = "in";
config const output = "out";

proc Usage(prog_name: string, ret_val: int)
{
	writeln("USAGE: ", prog_name, " --iterations=<iterations> --timestep=<timestep> --input=<input file> --output=<output file>\n");
	exit(ret_val);
}

proc main 
{
	//Usage("command", 1);

  writeln("iterations: ", iterations, " timestep: ", timestep, " input: ", input, " output: ", output, "\n");
	set_calculations_timestep(timestep);

	var infile = new file(input,FileAccessMode.read);
  infile.open();
  var num_bodies = body_get_num_from_file(infile);
  infile.close();
  writeln("num_bodies: ", num_bodies);
	//body_get_list_from_file(infile, &bodies, (unsigned*) &num_bodies);


	//for (c = 0; c < iterations; c++) {
	//
  //}

	//return 0;
}

