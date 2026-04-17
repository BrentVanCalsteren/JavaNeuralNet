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
    private double[][] dW;
    private double[] dB;

    private static final Random rand = new Random();

    public flat_layer(int inputSize, int outputSize, activation_fun fun) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.act_fun = fun;

        this.weights = new double[outputSize][inputSize];
        this.biases = new double[outputSize];

        initWeights();
    }

    private void initWeights() {
        //spread out 10 over all the weights Xavier method for tahn/sigmoid
        if (act_fun == activation_fun.SIGMOID | act_fun == activation_fun.TANH) {
            double limit = Math.sqrt(10.0 / (inputSize + outputSize));
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < inputSize; j++) {
                    weights[i][j] = rand.nextDouble() * 2 * limit - limit;
                }
                biases[i] = 0.0;
            }
        }else{
            double std = Math.sqrt(1.0 / inputSize);
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < inputSize; j++) {
                    weights[i][j] = rand.nextGaussian() * std;
                }
                biases[i] = 0.0;
            }
        }

    }

    public double[] forward(double[] input) {
        this.cache_lastInput = input.clone();

        double[] z = new double[this.outputSize];
        for (int i = 0; i < this.outputSize; i++) {
            double sum = this.biases[i];
            for (int j = 0; j < this.inputSize; j++) {
                sum += this.weights[i][j] * input[j];
            }
            z[i] = sum;
        }
        this.cache_lastZ = z.clone();
        this.cache_lastOutput = this.act_fun.activate_array(z);
        return this.cache_lastOutput.clone();
    }

    public double[] backward(double[] dOut) {
        double[] dZ = new double[this.outputSize];
        for (int i = 0; i < this.outputSize; i++) {
            dZ[i] = dOut[i] * act_fun.derive(this.cache_lastZ[i]);
        }

        double[][] dW = new double[this.outputSize][this.inputSize];
        double[] dB = new double[this.outputSize];
        for (int i = 0; i < this.outputSize; i++) {
            for (int j = 0; j < this.inputSize; j++) {
                dW[i][j] = dZ[i] * this.cache_lastInput[j];
            }
            dB[i] = dZ[i];
        }

        double[] dInput = new double[this.inputSize];
        for (int j = 0; j < this.inputSize; j++) {
            double sum = 0.0;
            for (int i = 0; i < this.outputSize; i++) {
                sum += this.weights[i][j] * dZ[i];
            }
            dInput[j] = sum;
        }

        this.dW = dW;
        this.dB = dB;

        return dInput;
    }

    public void updateParameters(double learningRate) {
        // Clip gradients to [-1, 1] for stability
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                double grad = dW[i][j];
                if (grad > 1.0) grad = 1.0;
                if (grad < -1.0) grad = -1.0;
                weights[i][j] -= learningRate * grad;
            }
            double gradB = dB[i];
            if (gradB > 1.0) gradB = 1.0;
            if (gradB < -1.0) gradB = -1.0;
            biases[i] -= learningRate * gradB;
        }
    }

}
