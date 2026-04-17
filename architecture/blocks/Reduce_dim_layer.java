package architecture.blocks;

public class Reduce_dim_layer extends Layer {
    private int input_depth;
    private int input_height;
    private int input_width;
    private int output_size;

    public Reduce_dim_layer(int[] size_dim) {
        this.input_depth = size_dim[0];
        this.input_height = size_dim[1];
        this.input_width = size_dim[2];
        this.output_size = this.input_depth * this.input_height * this.input_width;
    }

    public double[] forward_(double[][][] input) {
        double[] flat = new double[output_size];
        int idx = 0;
        for (int c = 0; c < input_depth; c++) {
            for (int h = 0; h < input_height; h++) {
                for (int w = 0; w < input_width; w++) {
                    flat[idx++] = input[c][h][w];
                }
            }
        }
        return flat;
    }

    public double[][][] backward_(double[] dOut) {
        double[][][] dVolume = new double[input_depth][input_height][input_width];
        int idx = 0;
        for (int c = 0; c < input_depth; c++) {
            for (int h = 0; h < input_height; h++) {
                for (int w = 0; w < input_width; w++) {
                    dVolume[c][h][w] = dOut[idx++];
                }
            }
        }
        return dVolume;
    }

    public int getOutput_size() { return output_size; }

    public void updateParameters(double lr) { }
}