
use Math;
//struct body_geom_t {
//	double x;			// x positions
//	double y;			// y positions
//	double mass;		// masses
//	double x_vel;		// x velocities
//	double y_vel;		// y velocities
//	double x_accel;		// used for force calculations
//	double y_accel;		// used for force calculations
//};
//typedef struct body_geom_t body_geom_t;

class body_geom_t {
  var x, y, mass, x_vel, y_vel, x_accel, y_accel: real;
}


//struct tree_node_t {
//	double diam;
//	double quad_x;
//	double quad_y;
//	body_geom_t g;
//	int q[4];
//	int parent;
//	int padding;
//};
//typedef struct tree_node_t tree_node_t;

class tree_node_t {
  var diam, quad_x, quad_y: real;
	var g: body_geom_t;
	var parent, padding: int;
  var q: [0..3] int;
};

//static double delta_time = 1.0;
var delta_time: real = 1.0;
var soften: real = 18.0;
var damping: real = 0.1;
var theta: real = 0.5;


// given 2 bodies, v and t, the force and the resultant accleration enforced by
// body v ON body t is given by the below function.
//
// from here:
// http://www.cs.utexas.edu/users/akanksha/cs380p/leapfrog.txt
//void compute_accln(body_geom_t const *const v, body_geom_t *t)
//{
//	double x = v->x;
//	double y = v->y;
//
//	x -= t->x;
//	y -= t->y;
//
//	double dist = x*x + y*y;
//	dist += SOFTEN;
//
//	dist = sqrt(dist * dist * dist);
//	double mag = v->mass / dist;
//
//	t->x_accel += mag * x;
//	t->y_accel += mag * y;
//}

proc compute_accln(v: body_geom_t, inout t: body_geom_t ) {

	var x: real = v.x;
	var y: real = v.y;

	x -= t.x;
	y -= t.y;

	var dist: real = x*x + y*y;
	dist += soften;

	dist = sqrt(dist * dist * dist);
	var mag: real = v.mass / dist;

	t.x_accel += mag * x;
	t.y_accel += mag * y;
}


// the below function is used to move the body ONLY after the resultant
// acceleration experienced by a body t by virtue of all other bodies is
// computed.
//
// from here:
// http://www.cs.utexas.edu/users/akanksha/cs380p/leapfrog.txt
//void move_body(body_geom_t *t) 
//{
//    // Notes : From NVIDIA CUDA SDL
//    // acceleration = force / mass;
//    // new velocity = old velocity + acceleration * deltaTime
//    // note: factor out the body's mass from the equation, here and in computeaccln
//    // (because they cancel out).  Thus here force == acceleration
//
//	// deltaTime -> is the duration of movement of each body during a single
//	// simulation of nbody system.
//    t->x_vel += t->x_accel * delta_time; 
//    t->y_vel += t->y_accel * delta_time;
//
//	// damping is used to control how much a body moves in free space
//    t->x_vel *= DAMPING; 
//    t->y_vel *= DAMPING;
//
//    t->x += t->x_vel * delta_time;
//    t->y += t->y_vel * delta_time;
//}
proc move_body(inout t: body_geom_t) 
{
    // Notes : From NVIDIA CUDA SDL
    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note: factor out the body's mass from the equation, here and in computeaccln
    // (because they cancel out).  Thus here force == acceleration

	// deltaTime -> is the duration of movement of each body during a single
	// simulation of nbody system.
    t.x_vel += t.x_accel * delta_time; 
    t.y_vel += t.y_accel * delta_time;

	// damping is used to control how much a body moves in free space
    t.x_vel *= damping; 
    t.y_vel *= damping;

    t.x += t.x_vel * delta_time;
    t.y += t.y_vel * delta_time;
}


// this is our multipole acceptance criterion for whether a the force on a
// particle "b" from another body/group of bodies "n" can be calculated with
// acceptable error
//
// this uses the simple form:
//
//		d > l / theta
//
// where "d" is the distance from "b" to center of mass of "n", and "l" is the
// length of the side of the square enclosing "n".  "theta" is a constant that
// should be acceptably set to 0.5
//
// there is supposed to be a more accurate form:
//
//		d > (1 / theta + sigma)
//
// where "sigma" is the distance from the center of mass of "n" to the
// geometric center.  see the "cell opening criterion" of "a parallel treecode"
// by john dubinski
//int MAC_acceptable(tree_node_t const *const n, body_geom_t const *const b)
//{
//	double x = b->x - n->g.x;
//	double y = b->y - n->g.y;
//
//	return sqrt(x*x + y*y) > (1.0 * n->diam / THETA);
//}
//
//void set_calculations_timestep(double timestep)
//{
//	delta_time = timestep;
//}
proc MAC_acceptable(n: tree_node_t, b: body_geom_t): int
{
	var x: real = b.x - n.g.x;
	var y: real = b.y - n.g.y;

	return sqrt(x*x + y*y) > (1.0 * n.diam / theta);
}

proc set_calculations_timestep(timestep: real)
{
	delta_time = timestep;
}
