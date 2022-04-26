class BHTree {
   private:

   public:
    BHTree();
    ~BHTree();

    void compute_nonedge_forces(float* points, int num_points);
};