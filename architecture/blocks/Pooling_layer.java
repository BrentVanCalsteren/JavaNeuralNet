package architecture.blocks;

public class Pooling_layer extends Layer {
    private int window_size;   // square pool window (e.g., 2)
    private int window_jump;

    // Cache for backprop (max positions)
    private int[][][][] maxIndices; // [channel][outH][outW][2] storing (row, col)

    public Pooling_layer(int[] window_size_jump) {
        this.window_size = window_size_jump[0];
        this.window_jump = window_size_jump[1];
    }

    public double[][][] forward(double[][][] input) {
        int channels = input.length;
        int inHeight = input[0].length;
        int inWidth = input[0][0].length;
        int outHeight = (inHeight - window_size) / window_jump + 1;
        int outWidth = (inWidth - window_size) / window_jump + 1;

        double[][][] output = new double[channels][outHeight][outWidth];
        maxIndices = new int[channels][outHeight][outWidth][2];

        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    double maxVal = Double.NEGATIVE_INFINITY;
                    int maxI = -1, maxJ = -1;
                    for (int i = 0; i < window_size; i++) {
                        for (int j = 0; j < window_size; j++) {
                            int inH = h * window_jump + i;
                            int inW = w * window_jump + j;
                            double val = input[c][inH][inW];
                            if (val > maxVal) {
                                maxVal = val;
                                maxI = inH;
                                maxJ = inW;
                            }
                        }
                    }
                    output[c][h][w] = maxVal;
                    maxIndices[c][h][w][0] = maxI;
                    maxIndices[c][h][w][1] = maxJ;
                }
            }
        }
        return output;
    }
    public double[][][] backward(double[][][] dOut) {
        int channels = dOut.length;
        int outHeight = dOut[0].length;
        int outWidth = dOut[0][0].length;
        int inHeight = maxIndices[0][0][0][0] * 0 + (outHeight - 1) * window_jump + window_size; // reconstruct
        int inWidth = (outWidth - 1) * window_jump + window_size;

        double[][][] dInput = new double[channels][inHeight][inWidth];

        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    int maxH = maxIndices[c][h][w][0];
                    int maxW = maxIndices[c][h][w][1];
                    dInput[c][maxH][maxW] += dOut[c][h][w];
                }
            }
        }
        return dInput;
    }

    public void updateParameters(double learningRate) { }
}