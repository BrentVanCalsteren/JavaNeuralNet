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
    public Object learn_from_input(Object input, Object target){
        Object output = get_output(input);
        Object loss = grad_and_loss.gradient(output, target);
        Object l = loss;
        for(int i = layers.size()-1; i>=0;i--) {
            Layer layer = layers.get(i);
            if (layer instanceof Flat_layer) {
                assert loss instanceof double[];
                loss = ((Flat_layer)layer).backward((double[]) loss);
            } else if (layer instanceof Reduce_dim_layer) {
                assert loss instanceof double[];
                loss = ((Reduce_dim_layer)layer).backward((double[]) loss);
            } else if (layer instanceof Conv_layer) {
                assert loss instanceof double[][][];
                loss = ((Conv_layer)layer).backward((double[][][]) loss);
            } else if (layer instanceof Pooling_layer) {
                assert loss instanceof double[][][];
                loss = ((Pooling_layer)layer).forward((double[][][]) loss);
            }
        }

        for(int i = layers.size()-1; i>=0;i--) {
            layers.get(i).updateParameters(learning_rate);
        }
        return l;
    }

    public Object get_output(Object input){
        if (layers.isEmpty()) {return null;}
        Object output = input;
        for (Layer layer : layers) {
            if (layer instanceof Flat_layer) {
                output = ((Flat_layer)layer).forward((double[]) output);
            } else if (layer instanceof Reduce_dim_layer) {
                output = ((Reduce_dim_layer)layer).forward((double[][][]) output);
            } else if (layer instanceof Conv_layer) {
                output = ((Conv_layer)layer).forward((double[][][]) output);
            } else if (layer instanceof Pooling_layer) {
                output = ((Pooling_layer)layer).forward((double[][][]) output);
            }
        }
        return output;

    }
}
