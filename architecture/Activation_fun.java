package architecture;

public enum Activation_fun {
    TANH {
        @Override
        public double activate(double x) {
            return Math.tanh(x);
        }
        @Override
        public double derive(double x) {
            double t = Math.tanh(x);
            return 1 - t * t;
        }
    },
    SIGMOID {
        @Override
        public double activate(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
        @Override
        public double derive(double x) {
            double s = activate(x);
            return s * (1 - s);
        }
    },
    RELU {
        @Override
        public double activate(double x) {
            return x > 0 ? x : 0;
        }
        @Override
        public double derive(double x) {
            return x > 0 ? 1.0 : 0.0;
        }
    },
    LINEAR {
        @Override
        public double activate(double x) { return x; }
        @Override
        public double derive(double x) { return 1.0; }
    }
    //ADD ACTIVATION FOR CNN


    ;

    public abstract double activate(double x);
    public abstract double derive(double x);


    public double[] activate_1D_array(double[] vec) {
        double[] out = new double[vec.length];
        for (int i = 0; i < vec.length; i++) out[i] = activate(vec[i]);
        return out;
    }

    public double[][][] activate_3D_array(double[][][] volume) {
        int depth = volume.length;
        int height = volume[0].length;
        int width = volume[0][0].length;
        double[][][] out = new double[depth][height][width];
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    out[d][h][w] = activate(volume[d][h][w]);
                }
            }
        }
        return out;
    }


    public double[][][] derive_3D_array(double[][][] volume) {
        int depth = volume.length;
        int height = volume[0].length;
        int width = volume[0][0].length;
        double[][][] out = new double[depth][height][width];
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    out[d][h][w] = derive(volume[d][h][w]);
                }
            }
        }
        return out;
    }

}
