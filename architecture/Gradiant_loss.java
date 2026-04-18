package architecture;

public enum Gradiant_loss {

    /**
     * Mean Squared Error for regression tasks.
     * Loss = 1/2 * sum((predicted_i - target_i)^2)
     * Gradient = predicted_i - target_i
     */
    MSE {
        @Override
        public double loss(double[] predicted, double[] target) {
            validateLengths(predicted, target);
            double sum = 0.0;
            for (int i = 0; i < predicted.length; i++) {
                double diff = predicted[i] - target[i];
                sum += diff * diff;
            }
            return 0.5 * sum;
        }

        @Override
        public double[] gradient_1D(double[] predicted, double[] target) {
            validateLengths(predicted, target);
            double[] grad = new double[predicted.length];
            for (int i = 0; i < predicted.length; i++) {
                grad[i] = predicted[i] - target[i];
            }
            return grad;
        }

        @Override
        public double[][] gradient_2D(double[][] predicted, double[][] target) {
            return new double[0][];
        }

        @Override
        public double[][][] gradient_3D(double[][][] predicted, double[][][] target) {
            return new double[0][][];
        }
    },

    /**
     * Binary Cross-Entropy (for binary/multi-label classification).
     * Loss = - sum( target_i * log(sigmoid(p_i)) + (1-target_i) * log(1 - sigmoid(p_i)) )
     * Gradient = sigmoid(predicted_i) - target_i
     */
    BINARY_CROSS_ENTROPY {
        @Override
        public double loss(double[] predicted, double[] target) {
            validateLengths(predicted, target);
            double sum = 0.0;
            for (int i = 0; i < predicted.length; i++) {
                double p = sigmoid(predicted[i]);
                // Clip probabilities to avoid log(0)
                p = Math.max(1e-15, Math.min(1 - 1e-15, p));
                sum += -target[i] * Math.log(p) - (1 - target[i]) * Math.log(1 - p);
            }
            return sum;
        }

        @Override
        public double[] gradient_1D(double[] predicted, double[] target) {
            validateLengths(predicted, target);
            double[] grad = new double[predicted.length];
            for (int i = 0; i < predicted.length; i++) {
                grad[i] = sigmoid(predicted[i]) - target[i];
            }
            return grad;
        }

        @Override
        public double[][] gradient_2D(double[][] predicted, double[][] target) {
            return new double[0][];
        }

        @Override
        public double[][][] gradient_3D(double[][][] predicted, double[][][] target) {
            return new double[0][][];
        }
    },

    /**
     * Categorical Cross-Entropy with Softmax (for multi-class classification).
     * Loss = - sum( target_i * log(softmax(predicted)_i) )
     * Gradient = softmax(predicted)_i - target_i
     */
    CATEGORICAL_CROSS_ENTROPY {
        public double loss(double[] predicted, int targetIndex) {
            double[] probs = softmax(predicted);
            return -Math.log(probs[targetIndex] + 1e-15);
        }


        public double[] gradient(double[] predicted, int targetIndex) {
            double[] probs = softmax(predicted);
            double[] grad = new double[predicted.length];
            for (int i = 0; i < predicted.length; i++) {
                grad[i] = probs[i];
            }
            grad[targetIndex] -= 1.0;
            return grad;
        }

        @Override
        public double loss(double[] predicted, double[] target) {
            validateLengths(predicted, target);
            double[] probs = softmax(predicted);
            double sum = 0.0;
            for (int i = 0; i < probs.length; i++) {
                // Add small epsilon to avoid log(0)
                sum += -target[i] * Math.log(probs[i] + 1e-15);
            }
            return sum;
        }

        @Override
        public double[] gradient_1D(double[] predicted, double[] target) {
            validateLengths(predicted, target);
            double[] probs = softmax(predicted);
            double[] grad = new double[predicted.length];
            for (int i = 0; i < predicted.length; i++) {
                grad[i] = probs[i] - target[i];
            }
            return grad;
        }

        @Override
        public double[][] gradient_2D(double[][] predicted, double[][] target) { //Todo: further implement
            return new double[0][];
        }

        @Override
        public double[][][] gradient_3D(double[][][] predicted, double[][][] target) {
            return new double[0][][];
        }
    },


    /**
     * Huber Loss (smooth L1) for robust regression.
     * Less sensitive to outliers than MSE.
     */
    HUBER {
        private double delta = 1.0;

        public void setDelta(double d) {
            this.delta = d;
        }

        @Override
        public double loss(double[] predicted, double[] target) {
            validateLengths(predicted, target);
            double sum = 0.0;
            for (int i = 0; i < predicted.length; i++) {
                double diff = Math.abs(predicted[i] - target[i]);
                if (diff <= delta) {
                    sum += 0.5 * diff * diff;
                } else {
                    sum += delta * (diff - 0.5 * delta);
                }
            }
            return sum;
        }

        @Override
        public double[] gradient_1D(double[] predicted, double[] target) {
            validateLengths(predicted, target);
            double[] grad = new double[predicted.length];
            for (int i = 0; i < predicted.length; i++) {
                double diff = predicted[i] - target[i];
                if (Math.abs(diff) <= delta) {
                    grad[i] = diff;
                } else {
                    grad[i] = delta * Math.signum(diff);
                }
            }
            return grad;
        }

        @Override
        public double[][] gradient_2D(double[][] predicted, double[][] target) {
            validateDimensions(predicted, target);
            int rows = predicted.length;
            int cols = predicted[0].length;
            double[][] grad = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double diff = predicted[i][j] - target[i][j];
                    if (Math.abs(diff) <= delta) {
                        grad[i][j] = diff;
                    } else {
                        grad[i][j] = delta * Math.signum(diff);
                    }
                }
            }
            return grad;
        }

        @Override
        public double[][][] gradient_3D(double[][][] predicted, double[][][] target) {
            validateDimensions(predicted, target);
            int d1 = predicted.length;
            int d2 = predicted[0].length;
            int d3 = predicted[0][0].length;
            double[][][] grad = new double[d1][d2][d3];
            for (int i = 0; i < d1; i++) {
                for (int j = 0; j < d2; j++) {
                    for (int k = 0; k < d3; k++) {
                        double diff = predicted[i][j][k] - target[i][j][k];
                        if (Math.abs(diff) <= delta) {
                            grad[i][j][k] = diff;
                        } else {
                            grad[i][j][k] = delta * Math.signum(diff);
                        }
                    }
                }
            }
            return grad;
        }
    };

    public abstract double loss(double[] predicted, double[] target);
    public abstract double[] gradient_1D(double[] predicted, double[] target);
    public abstract double[][] gradient_2D(double[][] predicted, double[][] target);
    public abstract double[][][] gradient_3D(double[][][] predicted, double[][][] target);
    /////////////////////////////
    //Utility fun
    /////////////////////////////
    protected static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    protected static double[] softmax(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logits) {
            if (v > max) max = v;
        }
        double sum = 0.0;
        double[] exp = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exp[i] = Math.exp(logits[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < logits.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }

    protected static void validateLengths(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Length mismatch: predicted=" + a.length + ", target=" + b.length);
        }
    }

    protected static void validateDimensions(double[][] a, double[][] b) {
        if (a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("2D shape mismatch");
        }
    }

    protected static void validateDimensions(double[][][] a, double[][][] b) {
        if (a.length != b.length || a[0].length != b[0].length || a[0][0].length != b[0][0].length) {
            throw new IllegalArgumentException("3D shape mismatch");
        }
    }
}

