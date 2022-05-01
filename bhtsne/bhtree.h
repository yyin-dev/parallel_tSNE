class BHTree {
   private:
    float num_points;

   public:
    BHTree(int num_points, float theta);
    ~BHTree();

    void compute_nonedge_forces(float* points, float* neg_forces, float* norm);
};