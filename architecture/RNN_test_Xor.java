package architecture;
import architecture.NN_types.rnn;
import architecture.blocks.Layer_data;
import architecture.blocks.Layer_type;

import java.util.Arrays;

public class RNN_test_Xor {
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
        int[] layer_output_size = {3, 6, 1};
        Layer_data[] layers= {
             new Layer_data(Layer_type.FLAT, 3,9,Activation_fun.RELU),
                new Layer_data(Layer_type.FLAT,9,1,Activation_fun.LINEAR)

        };
        rnn net = new rnn(layers, Gradiant_loss.BINARY_CROSS_ENTROPY,0.01);

        int epochs = 5000;
        for (int e = 0; e < epochs; e++) {
            double totalLoss = 0.0;
            for (int i = 0; i < inputs.length; i++) {
                totalLoss += net.learn_from_input(inputs[i], targets[i]);
            }
            if (e % 1000 == 0) {
                System.out.printf("Epoch %d, Avg Loss: %.6f%n", e, totalLoss / inputs.length);
            }
        }

        // Evaluate
        System.out.println("\nPredictions after training:");
        for (int i = 0; i < inputs.length; i++) {
            double[] pred = net.get_output(inputs[i]);
            double probability = Activation_fun.SIGMOID.activate(pred[0]); //only needed if output layer is LINEAR
            int predictedClass = probability > 0.5 ? 1 : 0;
            System.out.printf("Input: %s → Output: %.4f (class %d), Target: %.0f%n",
                    Arrays.toString(inputs[i]), probability, predictedClass, targets[i][0]);
        }
    }

}
