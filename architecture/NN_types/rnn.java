package architecture.NN_types;
import architecture.Activation_fun;
import architecture.blocks.*;
import architecture.Gradiant_loss;

import java.util.ArrayList;
import java.util.List;

public class rnn {
    private List<Flat_layer> layers = new ArrayList<>();
    private Gradiant_loss grad_and_loss;
    public double learning_rate = 0.01;

    public rnn(Layer_data[] data, Gradiant_loss grad_and_loss, double learning_rate) {
        this.grad_and_loss = grad_and_loss;
        this.learning_rate = learning_rate;
        this.generate_layers(data);
    }

    private void generate_layers(Layer_data[] data) {
        for(Layer_data d : data) {
            if(d.layer_type != Layer_type.FLAT) return;
            Flat_layer l = new Flat_layer(d.input_dim,d.output_dim,d.activation_function);
            layers.add(l);
        }

    }

    public double learn_from_input(double[] input,double[] target){
        double[] output = get_output(input);
        double[] dOutput = grad_and_loss.gradient(output, target);
        for(int i = layers.size()-1; i>=0;i--) {
            dOutput = layers.get(i).backward(dOutput);
        }

        for(int i = layers.size()-1; i>=0;i--) {
            layers.get(i).updateParameters(learning_rate);
        }
        return grad_and_loss.loss(output,target);
    }

    public double[] get_output(double[] input){
        double[] temp = layers.getFirst().forward(input);
        for(int i = 1; i<layers.size()-1;i++) {
            temp = layers.get(i).forward(temp);
        }
        return layers.getLast().forward(temp);
    }

}
