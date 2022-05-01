class BHTree {
   private:
    float num_points;

   public:
    BHTree(int num_points, float theta);
    ~BHTree();

    void compute_nonedge_forces(float* points);
    void get_nonedge_forces(float* neg_forces, float* sum_Q);
};