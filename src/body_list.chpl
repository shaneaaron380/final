
use classes;

// return the nubmer of bodies or ERROR on error
//int body_get_num_from_file(FILE *in)
//{
//	int r;
//
//	if (fscanf(in, "%d\n", &r) != 1)
//		return ERROR;
//
//	return r;
//}
proc body_get_num_from_file(infile: file): int
{
	var r = infile.read(int);

	return r;
}

// return ERROR on error, otherwise SUCCESS
//int body_get_from_file(FILE *in, body_t *b)
//{
//	if (fscanf(in, "%lf %lf %lf %lf %lf\n", &b->g.x, &b->g.y, &b->g.mass,
//				&b->g.x_vel, &b->g.y_vel) != 5)
//		return ERROR;
//
//	// these should be 0.0 before each iteration -- this may not be needed, but
//	// we'll do it just in case
//	b->g.x_accel = 0.0;
//	b->g.y_accel = 0.0;
//
//	return SUCCESS;
//}
proc body_get_from_file(infile: file, inout b: body_geom_t)
{

  b.x = infile.read(real);
  b.y = infile.read(real);
  b.mass = infile.read(real);
  b.x_vel = infile.read(real);
  b.y_vel = infile.read(real);

	// these should be 0.0 before each iteration -- this may not be needed, but
	// we'll do it just in case
	b.x_accel = 0.0;
	b.y_accel = 0.0;

}

// return ERROR or error, otherwise SUCCESS
//int body_get_list_from_file(const char *filename, body_t **bodies, unsigned
//		*num_bodies)
//{
//	FILE *in;
//	unsigned i;
//
//	if (! (in = fopen(filename, "r")))
//		RET_ERROR("could not open %s in body_get_list_from_file", filename);
//
//	if ((*num_bodies = (unsigned) body_get_num_from_file(in)) <= 0)
//		RET_ERROR("number of bodies was %u", *num_bodies);
//
//	if (! (*bodies = (body_t*) malloc(*num_bodies * sizeof(bodies[0][0]))))
//		RET_ERROR("could not allocate space for bodies");
////fprintf(stderr, "malloc'ing %lu bytes (%d * %lu) for bodies at %p\n", *num_bodies * sizeof(bodies[0][0]), *num_bodies, sizeof(bodies[0][0]), *bodies);
//
//	for (i = 0; i < *num_bodies; i++)
//		if (body_get_from_file(in, &(*bodies)[i]) != SUCCESS)
//			RET_ERROR("could not read in element %u", i);
//
//	fclose(in);
//
//	return SUCCESS;
//}
proc body_get_list_from_file(filename: string, inout bodies: body_geom_t, inout num_bodies: uint)
{
	var infile = new file(filename,FileAccessMode.read);

  infile.open();

	num_bodies = body_get_num_from_file(infile);

	for i in 0..num_bodies do
		body_get_from_file(infile, bodies[i]);

  infile.close();
}

//int body_get_num_from_filename(const char *filename, unsigned *num_bodies)
//{
//	FILE *in = NULL;
//
//	if (! (in = fopen(filename, "r")))
//		RET_ERROR("could not open %s in body_get_list_from_file", filename);
//
//	if ((*num_bodies = (unsigned) body_get_num_from_file(in)) <= 0)
//		RET_ERROR("number of bodies was %u", *num_bodies);
//
//	fclose(in);
//
//	return SUCCESS;
//}
proc body_get_num_from_filename(filename: string, inout num_bodies: uint)
{
	var infile = new file(filename,FileAccessMode.read);

  infile.open();

	num_bodies = body_get_num_from_file(infile);

  infile.close();
}

//int dump_bodies_to_file(char const *const fname, body_t const *const bodies,
//		unsigned num_bodies)
//{
//	unsigned i;
//	FILE *out;
//
//	if (! (out = fopen(fname, "w")))
//		RET_ERROR("could not open %s for writing", fname);
//
//	if (fprintf(out, "%u\n", num_bodies) <= 0)
//		RET_ERROR("could not write num_bodies out to file");
//
//	for (i = 0; i < num_bodies; i++) {
//		if (fprintf(out, "%lf %lf %lf %lf %lf\n", bodies[i].g.x, bodies[i].g.y,
//				bodies[i].g.mass, bodies[i].g.x_vel, bodies[i].g.y_vel) <= 0)
//			RET_ERROR("could not write particle %u out to file", i);
//		//fprintf(stderr, "%lf %lf %lf %lf %lf\n", bodies[i].g.x, bodies[i].g.y,
//		//        bodies[i].g.mass, bodies[i].g.x_vel, bodies[i].g.y_vel);
//	}
//
//	fclose(out);
//	
//	return SUCCESS;
//}
proc dump_bodies_to_file(filename: string, bodies: body_geom_t, num_bodies: uint)
{
	var outfile = new file(filename,FileAccessMode.write);

  outfile.open();

  outfile.writeln(num_bodies);

	for i in 0..num_bodies do
		outfile.writeln(bodies[i].x, " ", bodies[i].y, " ", bodies[i].mass, " ", bodies[i].x_vel, " ", bodies[i].y_vel);

  outfile.close();
}

