package architecture.RNN;
import architecture.activation_fun;
import architecture.blocks.*;
import architecture.gradiant_loss;

import java.util.ArrayList;
import java.util.List;

public class rnn {
    private List<flat_layer> layers;
    private int inputSize;
    private int outputSize;
    private gradiant_loss grad_and_loss;
    public double learning_rate = 0.01;

    public rnn(int[] sizes, gradiant_loss grad_and_loss) {
        this.inputSize = sizes[0];
        this.outputSize = sizes[sizes.length - 1];
        this.generate_layers(sizes);
    }

    private void generate_layers(int[] sizes) {
        layers = new ArrayList<>();
        for (int i = 1; i<sizes.length-1;i++) {
            layers.add(new flat_layer(sizes[i-1],sizes[i],activation_fun.TANH));
        }
        layers.add(new flat_layer(sizes[sizes.length-2],sizes[sizes.length-1],activation_fun.LINEAR));
    }

    public void learn_from_input(double[] input,double[] target){
        double[] output = get_output(input);
        double[] dOutput = grad_and_loss.gradient(output, target);
        for(int i = layers.size()-1; i>=0;i--) {
            dOutput = layers.get(i).backward(dOutput);
        }

        for(int i = layers.size()-1; i>=0;i--) {
            layers.get(i).updateParameters(learning_rate);
        }
    }

    public double[] get_output(double[] input){
        double[] temp = layers.getFirst().forward(input);
        for(int i = 1; i<layers.size()-1;i++) {
            temp = layers.get(i).forward(temp);
        }
        return layers.getLast().forward(temp);
    }




}
