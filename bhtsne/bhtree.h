class BHTree {
   private:

   public:
    BHTree(float *Ys, int num_points, int num_dims);
    ~BHTree();

    void compute_nonedge_forces(float theta);
};