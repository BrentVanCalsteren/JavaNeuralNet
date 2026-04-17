package architecture.blocks;

import architecture.Activation_fun;
import java.util.Random;

public class Conv_layer extends Layer {
    private int input_depth;
    private int input_height;
    private int input_width;
    private int num_kernels; //will try to capture patterns
    private int kernel_size; //square
    private int kernel_jump;
    private Activation_fun activation;

    // Weights: [filterIndex][inputChannel][row][col]
    private double[][][][] weights;
    // Biases: one per filter
    private double[] biases;

    // Cache for backprop
    private double[][][] lastInput;   // input volume
    private double[][][] lastZ;       // pre-activation volume
    private double[][][] lastOutput;  // after activation

    private double[][][][] dW;        // weight gradients
    private double[] dB;              // bias gradients

    private static final Random rand = new Random();

    public Conv_layer(int[] size_dim, int[] kernel_num_size_jump, Activation_fun activation) {
        this.input_depth = size_dim[0];
        this.input_height = size_dim[1];
        this.input_width = size_dim[2];
        this.num_kernels = kernel_num_size_jump[0];
        this.kernel_size = kernel_num_size_jump[1];
        this.kernel_jump = kernel_num_size_jump[2];
        this.activation = activation;

        // Initialize weights with He initialization (good for ReLU)
        double std = Math.sqrt(2.0 / (this.kernel_size^2 * this.input_depth));
        weights = new double[this.num_kernels][this.input_depth][this.kernel_size][this.kernel_size];
        for (int f = 0; f < this.num_kernels; f++) {
            for (int c = 0; c < this.input_depth; c++) {
                for (int i = 0; i < this.kernel_size; i++) {
                    for (int j = 0; j < this.kernel_size; j++) {
                        weights[f][c][i][j] = rand.nextGaussian() * std;
                    }
                }
            }
        }
        biases = new double[this.num_kernels]; // initialized to 0
    }

    public double[][][] forward(double[][][] input) {
        this.lastInput = copyVolume(input);

        int outHeight = (input_height - kernel_size) / kernel_jump + 1;
        int outWidth = (input_width - kernel_size) / kernel_jump + 1;

        double[][][] z = new double[num_kernels][outHeight][outWidth];

        // Convolution
        for (int f = 0; f < num_kernels; f++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    double sum = biases[f];
                    for (int c = 0; c < input_depth; c++) {
                        for (int i = 0; i < kernel_size; i++) {
                            for (int j = 0; j < kernel_size; j++) {
                                int inH = h * kernel_jump + i;
                                int inW = w * kernel_jump + j;
                                sum += weights[f][c][i][j] * input[c][inH][inW];
                            }
                        }
                    }
                    z[f][h][w] = sum;
                }
            }
        }

        this.lastZ = copyVolume(z);
        this.lastOutput = activation.activate_3D_array(z);
        return copyVolume(lastOutput);
    }

    public double[][][] backward(double[][][] dOut) {
        int outHeight = lastZ[0].length;
        int outWidth = lastZ[0][0].length;

        // dZ = dOut * activation'(Z)
        double[][][] dZ = new double[num_kernels][outHeight][outWidth];
        double[][][] deriv = activation.derive_3D_array(lastZ);
        for (int f = 0; f < num_kernels; f++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    dZ[f][h][w] = dOut[f][h][w] * deriv[f][h][w];
                }
            }
        }

        // Initialize gradients
        dW = new double[num_kernels][input_depth][kernel_size][kernel_size];
        dB = new double[num_kernels];
        double[][][] dInput = new double[input_depth][input_height][input_width];

        // Compute weight and bias gradients, and propagate to input
        for (int f = 0; f < num_kernels; f++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    double delta = dZ[f][h][w];
                    dB[f] += delta;

                    for (int c = 0; c < input_depth; c++) {
                        for (int i = 0; i < kernel_size; i++) {
                            for (int j = 0; j < kernel_size; j++) {
                                int inH = h * kernel_jump + i;
                                int inW = w * kernel_jump + j;
                                dW[f][c][i][j] += delta * lastInput[c][inH][inW];
                                dInput[c][inH][inW] += weights[f][c][i][j] * delta;
                            }
                        }
                    }
                }
            }
        }

        return dInput;
    }

    public void updateParameters(double learningRate) {
        for (int f = 0; f < num_kernels; f++) {
            for (int c = 0; c < input_depth; c++) {
                for (int i = 0; i < kernel_size; i++) {
                    for (int j = 0; j < kernel_size; j++) {
                        weights[f][c][i][j] -= learningRate * dW[f][c][i][j];
                    }
                }
            }
            biases[f] -= learningRate * dB[f];
        }
    }

    // Utility to deep copy a 3D volume
    private double[][][] copyVolume(double[][][] vol) {
        int d = vol.length;
        int h = vol[0].length;
        int w = vol[0][0].length;
        double[][][] copy = new double[d][h][w];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < h; j++) {
                System.arraycopy(vol[i][j], 0, copy[i][j], 0, w);
            }
        }
        return copy;
    }
}