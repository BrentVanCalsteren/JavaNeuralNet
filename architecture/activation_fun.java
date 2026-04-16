package architecture;

public enum activation_fun {
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
    };

    public abstract double activate(double x);
    public abstract double derive(double x);


    public double[] activate_array(double[] vec) {
        double[] out = new double[vec.length];
        for (int i = 0; i < vec.length; i++) out[i] = activate(vec[i]);
        return out;
    }
}
