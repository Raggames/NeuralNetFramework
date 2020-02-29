using System;


namespace NeuralNetwork.Scripts.NeuronActivatorFunctions
{
    public class ActivatorFactory
    {
        public static Activator Produce(ActivatorType type)
        {
            switch(type)
            {
                case ActivatorType.Sigmoid: return new LogisticActivator();
                case ActivatorType.Tanh: return new TanhActivator();
                case ActivatorType.Relu: return new ReluActivator();
                default: throw new ArgumentException("ActivatorTypeIsNotSupported");
            }
        }
    }
}
