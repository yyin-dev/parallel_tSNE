class BHTree {
   private:
    int num_points;

   public:
    BHTree(int num_points, float theta);
    ~BHTree();

    void points_to_device(float* points);
    void compute_nonedge_forces();
    void get_nonedge_forces(float* neg_forces, float* sum_Q);
};