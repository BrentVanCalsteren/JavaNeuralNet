package architecture;
import architecture.Activation_fun;
import architecture.Gradiant_loss;
import architecture.NN_types.cnn;
import architecture.NN_types.rnn;
import architecture.blocks.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CNN_mnist_test {

    private static final int INPUT_CHANNELS = 1;
    private static final int INPUT_HEIGHT = 28;
    private static final int INPUT_WIDTH = 28;
    private static final int NUM_CLASSES = 10;
    private static double train_test_split = 0.1;

    // Adjust these based on your available memory/time
    private static final int EPOCHS = 5;
    private static final double LEARNING_RATE = 0.01;

    public static void main(String[] args) {
        String csvPath = "mnist_flattened.csv";  // Update path if needed

        System.out.println("Loading MNIST data...");
        List<MNISTSample> allSamples = loadMNIST(csvPath);
        if (allSamples.isEmpty()) {
            System.err.println("No data loaded. Check file path.");
            return;
        }

        Collections.shuffle(allSamples);
        List<MNISTSample> trainData = allSamples.subList(0, (int) (allSamples.size()*(1-train_test_split)));
        List<MNISTSample> testData = allSamples.subList(trainData.size(), allSamples.size());

        System.out.println("Train samples: " + trainData.size());
        System.out.println("Test samples: " + testData.size());

        // Build CNN
        //Conv -> Pool -> conv -> pool -> reduce_dim -> flat -> flat
        Layer_data[] layers = {
                new Layer_data(Layer_type.CONV,new int[]{1, 28, 28},new int[]{4, 7, 1},Activation_fun.RELU),
                new Layer_data(Layer_type.POOL,new int[]{2,2}),
                new Layer_data(Layer_type.REDUCE_DIM,new int[]{4, 11, 11}),
                new Layer_data(Layer_type.FLAT, 484,128,Activation_fun.RELU),
                new Layer_data(Layer_type.FLAT, 128,10,Activation_fun.LINEAR)
        };
        cnn cnn_net = new cnn(layers, Gradiant_loss.CATEGORICAL_CROSS_ENTROPY, 0.01);
        System.out.println("Training CNN...");

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            Collections.shuffle(trainData);
            double totalLoss = 0.0;

            for (int i = 0; i < trainData.size(); i++) {
                MNISTSample sample = trainData.get(i);
                double[][][] input = sample.image;  // [1][28][28]
                double[] target = sample.oneHotLabel;

                double loss = argmax((double[])cnn_net.learn_from_input(input,target));

                totalLoss += loss;
                // Print progress
                if (i % 500 == 0 && i > 0) {
                    System.out.printf("Epoch %d | Batch %d/%d | Loss: %.4f%n",
                            epoch + 1, i, trainData.size(), loss);
                }
            }

            double avgLoss = totalLoss / trainData.size();
            System.out.printf("Epoch %d complete | Avg Loss: %.6f%n", epoch + 1, avgLoss);

            // Evaluate on test set
            int correct = 0;
            for (MNISTSample sample : testData) {
                double[] result = (double[]) cnn_net.get_output(sample.image);
                int predicted = argmax(result);
                if (predicted == sample.label) correct++;
            }
            double accuracy = 100.0 * correct / testData.size();
            System.out.printf("Test Accuracy: %d/%d (%.2f%%)\n\n", correct, testData.size(), accuracy);
        }

        System.out.println("Training finished.");
    }

    // ---------------------- Data Loading ----------------------
    private static List<MNISTSample> loadMNIST(String filePath) {
        List<MNISTSample> samples = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length != INPUT_HEIGHT * INPUT_WIDTH + 1) {
                    continue; // skip malformed rows
                }
                int label = Integer.parseInt(parts[0].trim());
                double[][][] image = new double[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH];
                int idx = 1;
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        int val = Integer.parseInt(parts[idx++].trim());
                        image[0][h][w] = val / 255.0;  // normalize to [0,1]
                    }
                }
                samples.add(new MNISTSample(label, image));
            }
        } catch (IOException e) {
            System.err.println("Error reading CSV: " + e.getMessage());
        }
        return samples;
    }

    private static int argmax(double[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }

    static class MNISTSample {
        int label;
        double[][][] image;
        double[] oneHotLabel;   // new field for one-hot encoding

        MNISTSample(int label, double[][][] image) {
            this.label = label;
            this.image = image;
            this.oneHotLabel = new double[NUM_CLASSES];
            this.oneHotLabel[label] = 1.0;
        }
    }
}