package architecture.blocks;

import architecture.activation_fun;

import java.util.Random;

public class flat_layer {
    private double[][] weights;  // [outputSize][inputSize]
    private double[] biases;     // [outputSize]
    private int inputSize;
    private int outputSize;
    private activation_fun act_fun;

    //Cache
    private double[] cache_lastInput;
    private double[] cache_lastZ;
    private double[] cache_lastOutput;

    private static final Random rand = new Random();

    public flat_layer(int inputSize, int outputSize, activation_fun fun) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.act_fun = fun;

        weights = new double[outputSize][inputSize];
        biases = new double[outputSize];

        initWeights();
    }

    private void initWeights() {
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = rand.nextDouble() + rand.nextDouble() + 0.01;
            }
            biases[i] = 0.0;   //init at 0
        }
    }

    public double[] forward(double[] input) {
        this.cache_lastInput = input.clone();

        double[] z = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = biases[i];
            for (int j = 0; j < inputSize; j++) {
                sum += weights[i][j] * input[j];
            }
            z[i] = sum;
        }
        this.cache_lastZ = z.clone();
        this.cache_lastOutput = act_fun.activate_array(z);
        return this.cache_lastOutput.clone();
    }
    private double[][] dW;
    private double[] dB;
    public double[] backward(double[] dOut) {
        double[] dZ = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            dZ[i] = dOut[i] * act_fun.derive(this.cache_lastZ[i]);
        }

        double[][] dW = new double[outputSize][inputSize];
        double[] dB = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                dW[i][j] = dZ[i] * this.cache_lastInput[j];
            }
            dB[i] = dZ[i];
        }

        double[] dInput = new double[inputSize];
        for (int j = 0; j < inputSize; j++) {
            double sum = 0.0;
            for (int i = 0; i < outputSize; i++) {
                sum += weights[i][j] * dZ[i];
            }
            dInput[j] = sum;
        }

        this.dW = dW;
        this.dB = dB;

        return dInput;
    }

    public void updateParameters(double learningRate) {
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] -= learningRate * dW[i][j];
            }
            biases[i] -= learningRate * dB[i];
        }
    }

    public double[] getWeightsFlattened() {
        double[] flat = new double[outputSize * inputSize];
        int idx = 0;
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                flat[idx++] = weights[i][j];
            }
        }
        return flat;
    }

    public double[] getBiases() {
        return biases.clone();
    }

    public int getInputSize() { return inputSize; }
    public int getOutputSize() { return outputSize; }
}
