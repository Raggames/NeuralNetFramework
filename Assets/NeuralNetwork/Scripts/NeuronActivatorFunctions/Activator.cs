namespace NeuralNetwork.Scripts.NeuronActivatorFunctions
{
    public abstract class Activator
    {
        public abstract double CalculateValue(double input);
        public abstract double CalculateDeriviative(double input);

        public abstract ActivatorType GetActivatorType();
    }

    public enum ActivatorType
    {
        Tanh,
        Relu,
        Sigmoid,
        Softmax,
    }
}