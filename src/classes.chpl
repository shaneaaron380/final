

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
