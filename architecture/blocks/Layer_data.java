package architecture.blocks;
import architecture.Activation_fun;


public class Layer_data {
    public Layer_data(int input_dim,int output_dim,Activation_fun activation_function,Layer_type layer_type){
        this.input_dim=input_dim;
        this.output_dim=output_dim;
        this.activation_function=activation_function;
        this.layer_type=layer_type;
    }
    public Layer_data(int[] window_size_jump,Layer_type layer_type){
        if(layer_type == Layer_type.POOL)
            this.window_size_jump=window_size_jump;
        else if (layer_type == Layer_type.REDUCE_DIM){
            this.size_dim = window_size_jump;
        }
        this.layer_type=layer_type;
    }
    public Layer_data(int[] size_dim,int[] kernel_num_size_jump,Activation_fun activation_function,Layer_type layer_type){
        this.size_dim=size_dim;
        this.kernel_num_size_jump=kernel_num_size_jump;
        this.activation_function=activation_function;
        this.layer_type=layer_type;
    }

    public Layer_type layer_type;
    public Activation_fun activation_function;
    //flat layer
    public int input_dim;
    public int output_dim;
    //for pool layer
    public int[] window_size_jump;
    //convo layer + reduce layer
    public int[] size_dim;
    public int[] kernel_num_size_jump;
}
