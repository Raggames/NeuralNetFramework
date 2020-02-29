namespace NeuralNetwork.Scripts.NeuronActivatorFunctions
{
    public abstract class Activator
    {
        public abstract double CalculateValue(double input);
        public abstract double CalculateDerivative(double input);

        public abstract ActivatorType GetActivatorType();
    }

    public enum ActivatorType
    {
        Identity,
        Tanh,
        Relu,
        Sigmoid,
        Softmax,
        
        
    }
}