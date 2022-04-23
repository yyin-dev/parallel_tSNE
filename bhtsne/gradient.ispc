// re-define float min as ISPC cannot include C lib headers
uniform const float FLT_MIN = 1.17549435e-38f;

// Compute positive forces based on its neighbors
export inline void updateEdgeForces2d(uniform int start, uniform int end,
                              uniform int ind1,
                              uniform int uniform inp_col_P[], uniform float uniform inp_val_P[], uniform float uniform Y[], uniform bool eval_error,
                              uniform float uniform pos_f[], uniform float * uniform P_i_sum, uniform float * uniform C) {
  float partialC = .0f;
  float partialP_i = .0f;
  float partialPos_f0 = 0.f;
  float partialPos_f1 = 0.f;

  foreach (i = start ... end) {
    float D = 0.f;
    float t;

    int ind2 = inp_col_P[i] * 2; // no_dims

    t = Y[ind1] - Y[ind2];
    D += t * t;
    t = Y[ind1 + 1] - Y[ind2 + 1];
    D += t * t;

    if (eval_error) {
      partialP_i += inp_val_P[i];
      partialC += inp_val_P[i] * log((inp_val_P[i] + FLT_MIN) / ((1.0 / (1.0 + D)) + FLT_MIN));
    }

    D = inp_val_P[i] / (1.0 + D);

    partialPos_f0 += D * (Y[ind1] - Y[ind2]);
    partialPos_f1 += D * (Y[ind1 + 1] - Y[ind2 + 1]);
  }

  pos_f[ind1] = reduce_add(partialPos_f0);
  pos_f[ind1 + 1] = reduce_add(partialPos_f1);

  *P_i_sum = reduce_add(partialP_i);
  *C = reduce_add(partialC);
}


// compute low-dim points' gradient
// no significant benefit
export inline void computeFinalGradient(uniform float dC[], uniform float pos_f[],
                                uniform float neg_f[], uniform float sum_Q,
                                uniform int len) {
  foreach (i = 0 ... len) {
      dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
  }
}

inline bool sign(float f) {
  return f < 0;
}

// update low-dim points using gradient descent with momentum
// no significant benefit
export inline void gradientDescent(uniform float gains[], uniform float dY[],
                            uniform float uY[], uniform float Y[],
                            uniform float learning_rate, uniform float momentum,
                            uniform int len) {
  foreach (i = 0 ... len) {
      // Update gains
      gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8 + .01);

      // Perform gradient update (with momentum and gains)
      uY[i] = momentum * uY[i] - learning_rate * gains[i] * dY[i];
      Y[i] = Y[i] + uY[i];
  }
}