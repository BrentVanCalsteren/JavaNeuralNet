package architecture;
import architecture.RNN.rnn;

import java.util.Arrays;

public class test_rnn {
    public static void main(String[] args) {
        testXOR();
    }


    public static void testXOR() {
        System.out.println("=== XOR Problem ===");

        // XOR dataset
        double[][] inputs = {
                {0, 0,0},
                {0, 1,0},
                {1, 0,0},
                {1, 1,0},
                {0, 0,1},
                {0, 1,1},
                {1, 0,1},
                {1, 1,1}
        };
        double[][] targets = {{0},{1},{1},{0},{1},{0},{0},{0}};

        int[] layerSizes = {3, 4, 1};
        rnn net = new rnn(layerSizes, gradiant_loss.BINARY_CROSS_ENTROPY);

        int epochs = 5000;
        for (int e = 0; e < epochs; e++) {
            double totalLoss = 0.0;
            for (int i = 0; i < inputs.length; i++) {
                net.learn_from_input(inputs[i], targets[i]);
                double[] pred = net.get_output(inputs[i]);
                totalLoss += gradiant_loss.BINARY_CROSS_ENTROPY.loss(pred, targets[i]);
            }
            if (e % 1000 == 0) {
                System.out.printf("Epoch %d, Avg Loss: %.6f%n", e, totalLoss / inputs.length);
            }
        }

        // Evaluate
        System.out.println("\nPredictions after training:");
        for (int i = 0; i < inputs.length; i++) {
            double[] pred = net.get_output(inputs[i]);
            double probability = activation_fun.SIGMOID.activate(pred[0]); //only needed if output layer is LINEAR
            int predictedClass = probability > 0.5 ? 1 : 0;
            System.out.printf("Input: %s → Output: %.4f (class %d), Target: %.0f%n",
                    Arrays.toString(inputs[i]), probability, predictedClass, targets[i][0]);
        }
    }

}
