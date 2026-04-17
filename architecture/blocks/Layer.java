package architecture.blocks;

public class Layer {

    public double[] backward(double[] dOutput) {return null;}
    public double[] forward(double[] dOutput) {return null;}
    public double[][][] forward(double[][][] input) {return null;}
    public double[][][] backward(double[][][] dOut) { return null; }
    public void updateParameters(double learningRate) {}
    public double[] forward_(double[][][] input){return null;}
    public double[][][] backward_(double[] dOut) {return null;}
}
