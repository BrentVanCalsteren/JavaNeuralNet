package architecture.blocks;

import architecture.Activation_fun;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

public class Flat_layer_matrix extends Layer {
    private SimpleMatrix weights;  // [outputSize x inputSize]
    private SimpleMatrix biases;   // [outputSize x 1]
    private int inputSize;
    private int outputSize;
    private Activation_fun act_fun;

    // Cache for backprop
    private SimpleMatrix cache_lastInput;   // [inputSize x 1]
    private SimpleMatrix cache_lastZ;       // [outputSize x 1] (pre-activation)
    private SimpleMatrix cache_lastOutput;  // [outputSize x 1] (post-activation)

    private SimpleMatrix dW;  // weight gradients
    private SimpleMatrix dB;  // bias gradients

    private static final Random rand = new Random();

    public Flat_layer_matrix(int inputSize, int outputSize, Activation_fun fun) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.act_fun = fun;

        initWeights();
    }

    private void initWeights() {
        if (act_fun == Activation_fun.SIGMOID || act_fun == Activation_fun.TANH) {
            // Xavier uniform initialization
            double limit = Math.sqrt(6.0 / (inputSize + outputSize));
            weights = SimpleMatrix.random_DDRM(outputSize, inputSize, -limit, limit, rand);
        } else {
            // He normal initialization for ReLU / Linear
            double std = Math.sqrt(2.0 / inputSize);
            weights = new SimpleMatrix(outputSize, inputSize);
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < inputSize; j++) {
                    weights.set(i, j, rand.nextGaussian() * std);
                }
            }
        }
        biases = new SimpleMatrix(outputSize, 1);  // initialized to zero
    }

    public double[] forward(double[] input) {
        // Convert input to column vector
        SimpleMatrix x = new SimpleMatrix(inputSize, 1, true, input);
        this.cache_lastInput = x.copy();

        // z = weights * x + biases
        SimpleMatrix z = weights.mult(x).plus(biases);
        this.cache_lastZ = z.copy();

        // Apply activation element-wise
        double[] zData = z.getDDRM().getData();
        double[] activated = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            activated[i] = act_fun.activate(zData[i]);
        }
        this.cache_lastOutput = new SimpleMatrix(outputSize, 1, true, activated);

        return activated.clone();
    }

    public double[] backward(double[] dOut) {
        // Convert incoming gradient to column vector
        SimpleMatrix dOutMat = new SimpleMatrix(outputSize, 1, true, dOut);

        // Derivative of activation function
        double[] zData = cache_lastZ.getDDRM().getData();
        double[] deriv = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            deriv[i] = act_fun.derive(zData[i]);
        }
        SimpleMatrix derivMat = new SimpleMatrix(outputSize, 1, true, deriv);

        // dZ = dOut ⊙ activation'(Z)
        SimpleMatrix dZ = dOutMat.elementMult(derivMat);

        // Compute gradients
        this.dW = dZ.mult(cache_lastInput.transpose());
        this.dB = dZ.copy();

        // Gradient with respect to input: weights^T * dZ
        SimpleMatrix dInput = weights.transpose().mult(dZ);
        return dInput.getDDRM().getData();
    }
    @Override
    public void updateParameters(double learningRate) {
        // Clip gradients to prevent explosion
        double maxGrad = 1.0;
        SimpleMatrix clippedDW = clipElements(dW, maxGrad);
        SimpleMatrix clippedDB = clipElements(dB, maxGrad);

        // Update weights and biases
        weights = weights.minus(clippedDW.scale(learningRate));
        biases = biases.minus(clippedDB.scale(learningRate));
    }

    /**
     * Helper method to clip all elements of a matrix to [-limit, limit].
     */
    private SimpleMatrix clipElements(SimpleMatrix mat, double limit) {
        SimpleMatrix result = new SimpleMatrix(mat.numRows(), mat.numCols());
        double[] data = mat.getDDRM().getData();
        double[] clipped = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            clipped[i] = Math.max(-limit, Math.min(limit, data[i]));
        }
        result.getDDRM().setData(clipped);
        return result;
    }

    // Optional: getters for debugging or saving/loading
    public SimpleMatrix getWeights() { return weights; }
    public SimpleMatrix getBiases() { return biases; }
}
