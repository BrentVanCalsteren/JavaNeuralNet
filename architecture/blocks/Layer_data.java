package architecture.blocks;
import architecture.Activation_fun;


public class Layer_data {

    /**
     * @param layer_type = Layer_type.FLAT
     */
    public Layer_data(Layer_type layer_type, int input_dim,int output_dim,Activation_fun activation_function){
        assert layer_type == Layer_type.FLAT;
        this.input_dim=input_dim;
        this.output_dim=output_dim;
        this.activation_function=activation_function;
        this.layer_type=layer_type;
    }

    /**
     * @param layer_type = Layer_type.POOL | Layer_type.REDUCE_DIM
     */
    public Layer_data(Layer_type layer_type, int[] window_size_jump){
        assert layer_type == Layer_type.POOL || layer_type == Layer_type.REDUCE_DIM;
        if(layer_type == Layer_type.POOL) this.window_size_jump=window_size_jump;
        else this.size_dim = window_size_jump;
        this.layer_type=layer_type;
    }

    /**
     * @param layer_type = Layer_type.CONV
     */
    public Layer_data(Layer_type layer_type, int[] size_dim,int[] kernel_num_size_jump,Activation_fun activation_function){
        assert layer_type == Layer_type.CONV;
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
