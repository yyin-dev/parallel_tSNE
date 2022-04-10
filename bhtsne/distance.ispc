export uniform float euclideanDistance(uniform float p0[], uniform float p1[], uniform int dim) {
  varying float partialSum = .0f;

  foreach (d = 0 ... dim) {
    float t = p0[d] - p1[d];
    partialSum += t * t;
  }

  return reduce_add(partialSum);
}