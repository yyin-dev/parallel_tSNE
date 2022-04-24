class GradientDescender {
   private:
    int step;
    int num_points;
    int num_nodes;

    // Host memory
    float *h_left, *h_right, *h_bottom, *h_top;

    // projected points
    float *h_mass;
    float *h_x, *h_y;    // x, y coordinates
    float *h_ax, *h_ay;  // accumulated forces

    float *h_output;  // host output array for visualization

    int *h_child;
    int *h_start;
    int *h_sorted;
    int *h_count;

    // Device memory
    float *d_left, *d_right, *d_bottom, *d_top;

    float *d_mass;
    float *d_x, *d_y;
    float *d_ax, *d_ay;

    int *d_index;
    int *d_child;
    int *d_start;
    int *d_sorted;
    int *d_count;

    int *d_mutex;

    float *d_output;  // device output array for visualization

   public:
    GradientDescender(float *Ys, int num_points, int num_dims);

    void compute_nonedge_forces(float theta);
};