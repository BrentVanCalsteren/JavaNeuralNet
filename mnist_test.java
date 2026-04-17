import architecture.RNN.rnn;               // Your network class
import architecture.gradiant_loss;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class mnist_test {

    // MNIST constants
    private static final int INPUT_SIZE = 784;  // 28x28 pixels
    private static final int OUTPUT_SIZE = 10;   // digits 0-9
    private static double train_test_split = 0.1;

    public static void main(String[] args) {
        String csvFilePath = "mnist_flattened.csv";  // Change if needed

        System.out.println("Loading MNIST data from " + csvFilePath + "...");
        List<MNISTSample> allSamples = loadMNIST(csvFilePath);
        if (allSamples.isEmpty()) {
            System.err.println("No data loaded. Check file path and format.");
            return;
        }

        // Shuffle and split into train/test
        Collections.shuffle(allSamples);
        List<MNISTSample> trainData = allSamples.subList(0, (int) (allSamples.size()*(1-train_test_split)));
        List<MNISTSample> testData = allSamples.subList(trainData.size(), allSamples.size());

        System.out.println("Train samples: " + trainData.size());
        System.out.println("Test samples: " + testData.size());

        //network: 784 -> 800 -> 10 (wiki layout)
        int[] layerSizes = {INPUT_SIZE, 800, OUTPUT_SIZE};
        rnn net = new rnn(layerSizes, gradiant_loss.CATEGORICAL_CROSS_ENTROPY);
        net.learning_rate = 0.01;

        // Train
        int epochs = 10;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            Collections.shuffle(trainData);  // Shuffle each epoch

            for (MNISTSample sample : trainData) {
                double[] input = sample.pixels; //data is normilized
                double[] target = oneHot(sample.label, OUTPUT_SIZE);

                double loss = net.learn_from_input(input, target);
                totalLoss += loss;

            }

            double avgLoss = totalLoss / trainData.size();
            System.out.printf("Epoch %d | Avg Loss: %.6f%n", epoch + 1, avgLoss);

            int correct = 0;
            for (MNISTSample sample : testData) {
                double[] output = net.get_output(sample.pixels);
                int predicted = argmax(output);
                if (predicted == sample.label) correct++;
            }
            double accuracy = 100.0 * correct / testData.size();
            System.out.printf("Test Accuracy: %d/%d (%.2f%%)\n", correct, testData.size(), accuracy);
        }

        System.out.println("Training complete.");
    }

    private static List<MNISTSample> loadMNIST(String filePath) {
        List<MNISTSample> samples = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int count = 0;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length != INPUT_SIZE + 1) {
                    System.err.println("Skipping row with unexpected length: " + parts.length);
                    continue;
                }
                int label = Integer.parseInt(parts[0].trim());
                double[] pixels = new double[INPUT_SIZE];
                for (int i = 0; i < INPUT_SIZE; i++) {
                    int val = Integer.parseInt(parts[i + 1].trim());
                    pixels[i] = val / 255.0;   // Normalize to [0,1]
                }
                samples.add(new MNISTSample(label, pixels));
                count++;
            }
        } catch (IOException e) {
            System.err.println("Error reading CSV: " + e.getMessage());
        }
        return samples;
    }

    private static double[] oneHot(int label, int numClasses) {
        double[] vec = new double[numClasses];
        vec[label] = 1.0;
        return vec;
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
        double[] pixels;
        MNISTSample(int label, double[] pixels) {
            this.label = label;
            this.pixels = pixels;
        }
    }
}
