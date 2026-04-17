package architecture.NN_types;
import architecture.blocks.*;
import architecture.Gradiant_loss;

import java.util.ArrayList;
import java.util.List;

public class cnn {
    private Gradiant_loss grad_and_loss;
    private List<Layer> layers = new ArrayList<>();
    private double learning_rate = 0.01;


    public cnn(Layer_data[] layerTypes,double learning_rate, Gradiant_loss grad_and_loss) {
        this.grad_and_loss = grad_and_loss;
        this.generate_layers(layerTypes);
        this.learning_rate = learning_rate;
    }

    private void generate_layers(Layer_data[] layerTypes) {
        for(Layer_data layerType : layerTypes) {
            if (layerType.layer_type == Layer_type.FLAT){
                Layer l = new Flat_layer(layerType.input_dim,layerType.output_dim, layerType.activation_function);
                layers.add(l);
            }
            else if (layerType.layer_type == Layer_type.REDUCE_DIM) {
                Layer l = new Reduce_dim_layer(layerType.size_dim);
                layers.add(l);
            }
            else if (layerType.layer_type == Layer_type.CONV) {
                Layer l = new Conv_layer(layerType.size_dim, layerType.kernel_num_size_jump, layerType.activation_function);
                layers.add(l);
            }
            else if (layerType.layer_type == Layer_type.POOL) {
                Layer l = new Pooling_layer(layerType.window_size_jump);
                layers.add(l);
            }
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
        if (layers.isEmpty()) {return null;}
        double[] temp = layers.getFirst().forward(input);
        for(int i = 1; i<layers.size()-1;i++) {
            temp = layers.get(i).forward(temp);
        }
        return layers.getLast().forward(temp);
    }
}
